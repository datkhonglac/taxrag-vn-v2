"""TaxRAG VN – điểm vào ứng dụng Streamlit.

Chạy:  streamlit run app.py
"""

import hmac

import streamlit as st

st.set_page_config(page_title="TaxRAG VN – Hỏi đáp chính sách thuế", page_icon="⚖️",
                   layout="wide", initial_sidebar_state="expanded")

from taxrag.services import get_settings  # noqa: E402
from taxrag.ui import brand, inject_css  # noqa: E402

inject_css()
settings = get_settings()
st.session_state.setdefault("is_admin", False)

pages = [
    st.Page("views/hoi_dap.py", title="Hỏi đáp chính sách", icon=":material/forum:", default=True),
    st.Page("views/tai_lieu_rieng.py", title="Phân tích tài liệu riêng", icon=":material/description:"),
    st.Page("views/huong_dan.py", title="Hướng dẫn và phạm vi", icon=":material/info:"),
]
if st.session_state.is_admin:
    pages.append(st.Page("views/quan_tri.py", title="Quản trị dữ liệu", icon=":material/admin_panel_settings:"))

# Ẩn menu mặc định (luôn nằm trên cùng) để đặt logo lên đầu thanh bên
nav = st.navigation(pages, position="hidden")

with st.sidebar:
    brand()
    for page in pages:
        st.page_link(page)  # dùng tiêu đề và biểu tượng của trang
    st.divider()
    if st.session_state.is_admin:
        if st.button("Đăng xuất quản trị", use_container_width=True):
            st.session_state.is_admin = False
            st.rerun()
    else:
        with st.popover("Đăng nhập quản trị", use_container_width=True):
            if not settings.admin_password:
                st.caption("Chưa cấu hình ADMIN_PASSWORD.")
            else:
                with st.form("admin_login", border=False):
                    pw = st.text_input("Mật khẩu", type="password")
                    if st.form_submit_button("Đăng nhập", use_container_width=True):
                        if hmac.compare_digest(pw, settings.admin_password):
                            st.session_state.is_admin = True
                            st.rerun()
                        else:
                            st.error("Mật khẩu không đúng.")

nav.run()
