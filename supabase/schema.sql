-- ============================================================
-- TaxRAG VN v3 – lược đồ cơ sở dữ liệu Supabase
-- Chạy toàn bộ file trong Supabase > SQL Editor.
-- Viết theo kiểu "if not exists" nên chạy được trên CSDL v2 đang có.
-- ============================================================

create extension if not exists vector;
create extension if not exists pgcrypto;

-- ------------------------------------------------------------
-- 1. Danh mục văn bản (mục 8 – định hướng GVHD)
-- ------------------------------------------------------------
create table if not exists documents (
    id uuid primary key default gen_random_uuid(),
    document_name text,
    document_number text not null,
    uploaded_at timestamptz default now()
);

alter table documents add column if not exists document_title text;
alter table documents add column if not exists document_type text;          -- Luật, Nghị định, Thông tư, Công văn...
alter table documents add column if not exists issuing_authority text;
alter table documents add column if not exists issued_date date;
alter table documents add column if not exists effective_date date;
alter table documents add column if not exists expiry_date date;
alter table documents add column if not exists legal_status text default 'Chưa xác minh';
alter table documents add column if not exists tax_categories text[] default '{}';
alter table documents add column if not exists applicable_subjects text;    -- đối tượng áp dụng
alter table documents add column if not exists source_url text;
alter table documents add column if not exists retrieval_date date default current_date;
alter table documents add column if not exists filename text;
alter table documents add column if not exists content_hash text;
alter table documents add column if not exists page_count int;
alter table documents add column if not exists chunk_count int;
alter table documents add column if not exists uploaded_by text;
alter table documents add column if not exists notes text;
alter table documents add column if not exists updated_at timestamptz default now();
-- processing: đang nạp; ready: dùng được. Ứng dụng chỉ đọc văn bản 'ready',
-- nhờ vậy lỗi giữa chừng khi nạp không làm mất bản cũ.
alter table documents add column if not exists ingest_status text default 'ready';

-- Chuyển dữ liệu v2 (topic/status) sang cột mới nếu còn
do $$
begin
  if exists (select 1 from information_schema.columns
             where table_name = 'documents' and column_name = 'topic') then
    update documents set tax_categories = array[topic]
     where (tax_categories is null or tax_categories = '{}') and topic is not null;
  end if;
  if exists (select 1 from information_schema.columns
             where table_name = 'documents' and column_name = 'status') then
    update documents set legal_status = status
     where legal_status is null or legal_status = 'Chưa xác minh';
  end if;
end $$;
update documents set document_title = document_name where document_title is null;

create index if not exists documents_number_idx on documents (lower(document_number));

-- ------------------------------------------------------------
-- 2. Đoạn văn bản
-- ------------------------------------------------------------
create table if not exists chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid references documents(id) on delete cascade,
    chunk_index int,
    content text
);

alter table chunks add column if not exists page_number int;
alter table chunks add column if not exists page_end int;
alter table chunks add column if not exists strategy text default 'structural';
alter table chunks add column if not exists level text;             -- article | clause | point | ...
alter table chunks add column if not exists is_leaf boolean default true;
alter table chunks add column if not exists parent_id uuid;
alter table chunks add column if not exists heading text;
alter table chunks add column if not exists context text;
alter table chunks add column if not exists path text;
alter table chunks add column if not exists chapter text;
alter table chunks add column if not exists section text;
alter table chunks add column if not exists article text;
alter table chunks add column if not exists article_title text;
alter table chunks add column if not exists clause text;
alter table chunks add column if not exists point text;
alter table chunks add column if not exists covers text[] default '{}';
-- Số chiều phải khớp EMBEDDING_DIM (mặc định 768)
alter table chunks add column if not exists embedding vector(768);

-- Đánh dấu đoạn cũ của v2 (chia theo trang) để admin biết cần nạp lại
update chunks set strategy = 'legacy_page', level = 'legacy' where level is null;

-- Bảo đảm xóa văn bản thì xóa luôn các đoạn
alter table chunks drop constraint if exists chunks_document_id_fkey;
alter table chunks add constraint chunks_document_id_fkey
    foreign key (document_id) references documents(id) on delete cascade;

create index if not exists chunks_document_idx on chunks (document_id);
create index if not exists chunks_embedding_idx on chunks
    using hnsw (embedding vector_cosine_ops);

-- ------------------------------------------------------------
-- 3. Quan hệ giữa văn bản (sửa đổi / thay thế / hướng dẫn)
--    Lưu theo số hiệu để không mất liên kết khi nạp lại văn bản.
-- ------------------------------------------------------------
create table if not exists document_relations (
    id uuid primary key default gen_random_uuid(),
    source_number text not null,     -- văn bản sửa đổi / thay thế / hướng dẫn
    target_number text not null,     -- văn bản bị sửa đổi / bị thay thế / được hướng dẫn
    relation_type text not null check (relation_type in ('amends', 'replaces', 'guides')),
    note text,
    created_at timestamptz default now(),
    unique (source_number, target_number, relation_type)
);

-- ------------------------------------------------------------
-- 4. Lịch sử câu hỏi + đánh giá của người dùng (mục 22–23)
-- ------------------------------------------------------------
create table if not exists query_logs (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz default now(),
    session_id text,
    question text not null,
    as_of_label text,
    status text,
    answer jsonb,
    sources jsonb,
    config jsonb,
    latency_ms int,
    feedback smallint,          -- 1 = hữu ích, 0 = không hữu ích
    feedback_note text
);
create index if not exists query_logs_created_idx on query_logs (created_at desc);

-- ------------------------------------------------------------
-- 5. Tìm kiếm vector trên máy chủ (pgvector)
-- ------------------------------------------------------------
create or replace function match_chunks(
    query_embedding vector(768),
    match_count int default 40,
    doc_ids uuid[] default null
)
returns table (id uuid, similarity float)
language sql stable
as $$
    select c.id, 1 - (c.embedding <=> query_embedding) as similarity
    from chunks c
    join documents d on d.id = c.document_id
    where c.embedding is not null
      and coalesce(c.is_leaf, true)
      and d.ingest_status = 'ready'
      and (doc_ids is null or c.document_id = any(doc_ids))
    order by c.embedding <=> query_embedding
    limit match_count;
$$;

-- Ghi chú bảo mật: ứng dụng dùng Secret key (sb_secret_...) ở phía máy chủ Streamlit,
-- khóa này bỏ qua RLS. Không bao giờ đưa Secret key vào mã nguồn hoặc trình duyệt.
-- Bật RLS để chặn truy cập bằng publishable/anon key:
alter table documents enable row level security;
alter table chunks enable row level security;
alter table document_relations enable row level security;
alter table query_logs enable row level security;
