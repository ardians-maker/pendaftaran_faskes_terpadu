import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Konfigurasi Halaman
st.set_page_config(page_title="Sistem Faskes Terpadu", page_icon="🏥", layout="wide")

# ==========================================
# 1. SETUP DATABASE (EXCEL) - VERSI 4
# ==========================================
DB_PASIEN = 'db_pasien_v4.xlsx'
DB_ANTRIAN = 'db_antrian_v4.xlsx'
DB_DOKTER = 'db_dokter_v4.xlsx'
DB_REKAM_MEDIS = 'db_rekam_medis_v4.xlsx'
DB_FASILITAS = 'db_fasilitas_v4.xlsx'

# Daftar Faskes Global agar seragam di semua menu
DAFTAR_FASKES = [
    "Puskesmas A", "Puskesmas B", 
    "Klinik A", "Klinik B", 
    "RSUD A", "RSUD B", 
    "RSUP A", "RSUP B", 
    "RSND A", "RSND B"
]

def init_db(file_name, columns):
    if not os.path.exists(file_name):
        pd.DataFrame(columns=columns).to_excel(file_name, index=False)

init_db(DB_PASIEN, ['ID', 'Nama', 'Alamat', 'Face_Embedding'])
init_db(DB_ANTRIAN, ['Tanggal', 'ID_Antrian', 'Nama_Pasien', 'Faskes', 'Jenis', 'Poli', 'Keluhan', 'Status_Rujukan', 'Status_Periksa'])
# UPDATE: Penambahan kolom Nama_Dokter
init_db(DB_DOKTER, ['Nama_Dokter', 'Username', 'Password', 'Faskes', 'Poli', 'Status_Praktek'])
init_db(DB_REKAM_MEDIS, ['Tanggal', 'Nama_Pasien', 'Faskes', 'Diagnosis', 'Resep_Obat', 'Surat_Sakit', 'Rujukan_Tujuan'])
init_db(DB_FASILITAS, ['Faskes', 'Kapasitas_Rawat_Inap', 'Terisi', 'Status_Penuh'])

# ==========================================
# 2. LOAD MODEL PENDETEKSI WAJAH (.pkl)
# ==========================================


@st.cache_resource
def load_model():
    # Pastikan nama file di bawah ini sama persis dengan yang ada di GitHub Anda
    with open('best_face_recognition_model.pkl', 'rb') as file:
        return pickle.load(file)

face_model = load_model()

def ekstrak_fitur_wajah(image_buffer):
    return str(np.random.rand(128).tolist()) 

def verifikasi_wajah(image_buffer):
    df = pd.read_excel(DB_PASIEN)
    if df.empty:
        return False, "", ""
    return True, df.iloc[0]['Nama'], df.iloc[0]['Alamat']

# ==========================================
# 3. INISIALISASI SESSION STATE
# ==========================================
if "pasien_verified" not in st.session_state:
    st.session_state.pasien_verified = False
    st.session_state.p_nama = ""
    st.session_state.p_alamat = ""

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
    st.session_state.admin_faskes = ""

if "dokter_logged_in" not in st.session_state:
    st.session_state.dokter_logged_in = False
    st.session_state.nama_dokter = ""
    st.session_state.faskes_dokter = ""

def get_today_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

# ==========================================
# 4. TAMPILAN ANTARMUKA (UI)
# ==========================================
st.title("🏥 Sistem Informasi Terpadu Faskes & Rujukan")

tab_pasien, tab_admin, tab_dokter = st.tabs(["🧑‍🤝‍🧑 Area Pasien", "⚙️ Administrasi Faskes", "👨‍⚕️ Area Dokter"])

# ---------------------------------
# TAB 1: PASIEN
# ---------------------------------
with tab_pasien:
    menu_pasien = st.radio("Pilih Menu:", ["Ambil Antrian Berobat", "Riwayat Rekam Medis & Resep", "Daftar Akun Baru (Scan Wajah)"], horizontal=True)
    st.divider()

    if menu_pasien == "Daftar Akun Baru (Scan Wajah)":
        st.subheader("Pendaftaran Akun Baru")
        nama_input = st.text_input("Nama Lengkap")
        alamat_input = st.text_area("Alamat Lengkap")
        cam_daftar = st.camera_input("Ambil Foto Wajah untuk Pendaftaran")
        
        if st.button("Daftar Akun", type="primary"):
            if cam_daftar is None or not nama_input or not alamat_input:
                st.error("Gagal: Mohon isi nama, alamat, dan ambil foto wajah.")
            else:
                fitur = ekstrak_fitur_wajah(cam_daftar)
                df = pd.read_excel(DB_PASIEN)
                new_data = pd.DataFrame({'ID': [len(df)+1], 'Nama': [nama_input], 'Alamat': [alamat_input], 'Face_Embedding': [fitur]})
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_excel(DB_PASIEN, index=False)
                st.success(f"Berhasil: Pasien {nama_input} terdaftar. Silakan pindah ke menu Antrian.")

    elif menu_pasien == "Ambil Antrian Berobat" or menu_pasien == "Riwayat Rekam Medis & Resep":
        if not st.session_state.pasien_verified:
            st.subheader("Autentikasi Wajah")
            cam_login = st.camera_input("Scan Wajah untuk Melanjutkan")
            if st.button("Verifikasi Wajah", type="primary"):
                if cam_login is None:
                    st.error("Mohon scan wajah terlebih dahulu.")
                else:
                    status, nama, alamat = verifikasi_wajah(cam_login)
                    if status:
                        st.session_state.pasien_verified = True
                        st.session_state.p_nama = nama
                        st.session_state.p_alamat = alamat
                        st.rerun()
                    else:
                        st.error("Login Gagal: Wajah tidak dikenali.")
        else:
            if menu_pasien == "Ambil Antrian Berobat":
                st.subheader("Pengambilan Antrian")
                st.info(f"**Biodata Pasien:** {st.session_state.p_nama} ({st.session_state.p_alamat})")
                
                with st.expander("Lihat Jadwal Dokter & Info Kamar Rawat Inap", expanded=False):
                    df_dok_view = pd.read_excel(DB_DOKTER)
                    if not df_dok_view.empty:
                        # Menampilkan Nama Dokter di jadwal
                        st.dataframe(df_dok_view[['Faskes', 'Poli', 'Nama_Dokter', 'Status_Praktek']], use_container_width=True, hide_index=True)
                    df_fas_view = pd.read_excel(DB_FASILITAS)
                    if not df_fas_view.empty:
                        st.dataframe(df_fas_view, use_container_width=True, hide_index=True)
                
                faskes_pilih = st.selectbox("Pilih Faskes Tujuan", DAFTAR_FASKES)
                jenis_pasien = st.radio("Jenis Pasien", ["Biasa", "Rujukan"])
                layanan_pilih = st.selectbox("Pilih Layanan", ["Poli Umum", "Poli Gigi", "Poli Mata"])
                keluhan_input = st.text_area("Keluhan Singkat")
                
                if st.button("Ambil Antrian", type="primary"):
                    catatan_rujukan = ""
                    lanjut_antri = True
                    
                    if jenis_pasien == "Rujukan":
                        df_rm = pd.read_excel(DB_REKAM_MEDIS)
                        rujukan = df_rm[(df_rm['Nama_Pasien'] == st.session_state.p_nama) & (df_rm['Rujukan_Tujuan'] == faskes_pilih)]
                        if not rujukan.empty:
                            catatan_rujukan = f"[Rujukan dari {rujukan.iloc[-1]['Faskes']}]"
                            layanan_pilih = "Poli Rujukan"
                        else:
                            st.error("Rujukan tidak ditemukan.")
                            lanjut_antri = False
                    if lanjut_antri:
                        df_antrian = pd.read_excel(DB_ANTRIAN)
    
                        # PERBAIKAN 1: Ambil data antrian khusus untuk Faskes yang dipilih saja
                        df_faskes_ini = df_antrian[df_antrian['Faskes'] == faskes_pilih]
    
                        # PERBAIKAN 2: Nomor antrian dihitung hanya dari total pasien di Faskes tersebut
                        no_antrian = len(df_faskes_ini) + 1
    
                        # PERBAIKAN 3: Hitung jumlah orang yang belum diperiksa di Poli yang sama
                        antrian_sebelumnya = len(df_faskes_ini[(df_faskes_ini['Poli'] == layanan_pilih) & (df_faskes_ini['Status_Periksa'] == 'Belum')])
    
                        estimasi_menit = antrian_sebelumnya * 15
                        
                        new_data = pd.DataFrame({
                            'Tanggal': [get_today_str()], 'ID_Antrian': [no_antrian], 'Nama_Pasien': [st.session_state.p_nama], 
                            'Faskes': [faskes_pilih], 'Jenis': [jenis_pasien], 'Poli': [layanan_pilih], 
                            'Keluhan': [keluhan_input], 'Status_Rujukan': [catatan_rujukan], 'Status_Periksa': ['Belum']
                        })
                        df_antrian = pd.concat([df_antrian, new_data], ignore_index=True)
                        df_antrian.to_excel(DB_ANTRIAN, index=False)
                        
                        st.success("Berhasil!")
                        st.warning(f"🏥 **Nomor Antrian: {no_antrian}**\n\nMenunggu {antrian_sebelumnya} orang lagi. Estimasi dipanggil: **{estimasi_menit} menit** lagi.")
                        
            elif menu_pasien == "Riwayat Rekam Medis & Resep":
                st.subheader("Buku Kesehatan Pribadi (EMR)")
                df_rm = pd.read_excel(DB_REKAM_MEDIS)
                riwayat_ku = df_rm[df_rm['Nama_Pasien'] == st.session_state.p_nama]
                if riwayat_ku.empty:
                    st.info("Belum ada riwayat pemeriksaan.")
                else:
                    st.dataframe(riwayat_ku[['Tanggal', 'Faskes', 'Diagnosis', 'Resep_Obat', 'Surat_Sakit']], use_container_width=True, hide_index=True)

            if st.button("Keluar (Selesai)"):
                st.session_state.pasien_verified = False
                st.rerun()

# ---------------------------------
# TAB 2: ADMIN FASKES
# ---------------------------------
with tab_admin:
    # UPDATE: Database Akun Admin untuk seluruh faskes
    akun_admin = {
        "admin_puskesmasA": {"pwd": "sehatceria123", "faskes": "Puskesmas A"},
        "admin_puskesmasB": {"pwd": "sehatceria123", "faskes": "Puskesmas B"},
        "admin_klinikA": {"pwd": "sehatceria123", "faskes": "Klinik A"},
        "admin_klinikB": {"pwd": "sehatceria123", "faskes": "Klinik B"},
        "admin_rsudA": {"pwd": "sehatceria123", "faskes": "RSUD A"},
        "admin_rsudB": {"pwd": "sehatceria123", "faskes": "RSUD B"},
        "admin_rsupA": {"pwd": "sehatceria123", "faskes": "RSUP A"},
        "admin_rsupB": {"pwd": "sehatceria123", "faskes": "RSUP B"},
        "admin_rsndA": {"pwd": "sehatceria123", "faskes": "RSND A"},
        "admin_rsndB": {"pwd": "sehatceria123", "faskes": "RSND B"}
    }

    if not st.session_state.admin_logged_in:
        colA, colB, colC = st.columns([1, 2, 1])
        with colB:
            st.subheader("Login Sistem Administrasi")
            u_admin = st.text_input("Username Admin")
            p_admin = st.text_input("Password Admin", type="password")
            if st.button("Login"):
                if u_admin in akun_admin and p_admin == akun_admin[u_admin]["pwd"]:
                    st.session_state.admin_logged_in = True
                    st.session_state.admin_faskes = akun_admin[u_admin]["faskes"]
                    st.rerun()
                else:
                    st.error("Username atau Password Salah!")
    else:
        admin_faskes = st.session_state.admin_faskes
        col1, col2 = st.columns([4, 1])
        col1.subheader(f"⚙️ Panel Manajemen - {admin_faskes}")
        if col2.button("Logout Admin"):
            st.session_state.admin_logged_in = False
            st.rerun()
            
        st.divider()
        menu_admin = st.radio("Navigasi:", ["Dashboard & Antrian", "Manajemen Fasilitas & Dokter"], horizontal=True)
        
        if menu_admin == "Dashboard & Antrian":
            colL, colR = st.columns([1, 2])
            
            df_antri = pd.read_excel(DB_ANTRIAN)
            df_ku = df_antri[df_antri['Faskes'] == admin_faskes]
            
            with colL:
                st.markdown("**Statistik Poli Hari Ini**")
                if not df_ku.empty:
                    chart_data = df_ku['Poli'].value_counts()
                    st.bar_chart(chart_data)
                else:
                    st.caption("Belum ada data kunjungan.")
                    
            with colR:
                st.markdown("**Pemanggilan Antrian**")
                antrian_belum = df_ku[df_ku['Status_Periksa'] == 'Belum']
                if not antrian_belum.empty:
                    panggil = st.selectbox("Pilih Pasien untuk Dipanggil:", antrian_belum['Nama_Pasien'].tolist())
                    if st.button("Panggil & Ubah Status", type="primary"):
                        idx = df_antri.index[(df_antri['Nama_Pasien'] == panggil) & (df_antri['Status_Periksa'] == 'Belum')].tolist()[0]
                        df_antri.at[idx, 'Status_Periksa'] = 'Sedang Diperiksa'
                        df_antri.to_excel(DB_ANTRIAN, index=False)
                        st.success(f"Panggilan diteruskan ke dokter: {panggil}")
                        st.rerun()
                else:
                    st.info("Semua antrian sudah terpanggil/kosong.")
                
                st.dataframe(df_ku, use_container_width=True, hide_index=True)
                
        elif menu_admin == "Manajemen Fasilitas & Dokter":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Pendaftaran Dokter Baru**")
                with st.form("fdok"):
                    # UPDATE: Input tambahan Nama Dokter
                    nama_dok = st.text_input("Nama Lengkap Dokter (Contoh: dr. Andi Sp.PD)")
                    u_dok = st.text_input("Username (Untuk Login Dokter)")
                    p_dok = st.text_input("Password", type="password")
                    poli_dok = st.selectbox("Poli Penugasan", ["Poli Umum", "Poli Gigi", "Poli Mata"])
                    status_dok = st.selectbox("Status Praktek", ["Aktif", "Cuti"])
                    
                    if st.form_submit_button("Simpan Data"):
                        if not nama_dok or not u_dok:
                            st.warning("Nama Lengkap dan Username wajib diisi!")
                        else:
                            df = pd.read_excel(DB_DOKTER)
                            new_data = pd.DataFrame({'Nama_Dokter': [nama_dok], 'Username': [u_dok], 'Password': [p_dok], 'Faskes': [admin_faskes], 'Poli': [poli_dok], 'Status_Praktek': [status_dok]})
                            df = pd.concat([df, new_data], ignore_index=True)
                            df.to_excel(DB_DOKTER, index=False)
                            st.success(f"{nama_dok} berhasil didaftarkan!")
            with col2:
                st.markdown("**Sistem Kapasitas Rawat Inap**")
                df_fasilitas = pd.read_excel(DB_FASILITAS)
                current_fas = df_fasilitas[df_fasilitas['Faskes'] == admin_faskes]
                
                kapasitas = current_fas['Kapasitas_Rawat_Inap'].values[0] if not current_fas.empty else 0
                terisi = current_fas['Terisi'].values[0] if not current_fas.empty else 0
                
                st.info(f"Kapasitas: **{kapasitas}** Bed | Terisi: **{terisi}** Bed")
                
                with st.form("ffas"):
                    new_kapasitas = st.number_input("Update Kapasitas Total Bed", min_value=0, value=int(kapasitas))
                    new_terisi = st.number_input("Update Bed Terisi Saat Ini (Tambah/Kurangi)", min_value=0, max_value=new_kapasitas, value=int(terisi))
                    if st.form_submit_button("Update Sistem Bed"):
                        status_penuh = "Penuh" if new_terisi >= new_kapasitas else "Tersedia"
                        df_fasilitas = df_fasilitas[df_fasilitas['Faskes'] != admin_faskes]
                        new_df = pd.DataFrame({'Faskes': [admin_faskes], 'Kapasitas_Rawat_Inap': [new_kapasitas], 'Terisi': [new_terisi], 'Status_Penuh': [status_penuh]})
                        df_fasilitas = pd.concat([df_fasilitas, new_df], ignore_index=True)
                        df_fasilitas.to_excel(DB_FASILITAS, index=False)
                        st.success("Sistem Bed Diperbarui!")

# ---------------------------------
# TAB 3: DOKTER
# ---------------------------------
with tab_dokter:
    if not st.session_state.dokter_logged_in:
        colA, colB, colC = st.columns([1, 2, 1])
        with colB:
            st.subheader("Login Dokter")
            u_login = st.text_input("Username Dokter")
            p_login = st.text_input("Password Dokter", type="password")
            if st.button("Masuk"):
                df_dok = pd.read_excel(DB_DOKTER)
                dok_match = df_dok[(df_dok['Username'] == u_login) & (df_dok['Password'] == p_login)]
                if not dok_match.empty:
                    st.session_state.dokter_logged_in = True
                    # Mengambil Nama Dokter langsung dari database (bukan username lagi)
                    st.session_state.nama_dokter = dok_match['Nama_Dokter'].values[0] 
                    st.session_state.faskes_dokter = dok_match['Faskes'].values[0]
                    st.rerun()
                else:
                    st.error("Kredensial tidak valid.")
    else:
        col1, col2 = st.columns([4, 1])
        # Akan memunculkan "Workspace Medis: dr. Budi" dsb
        col1.success(f"Workspace Medis: **{st.session_state.nama_dokter}** ({st.session_state.faskes_dokter})")
        if col2.button("Logout"):
            st.session_state.dokter_logged_in = False
            st.rerun()
            
        df_antrian_dok = pd.read_excel(DB_ANTRIAN)
        pasien_tunggu = df_antrian_dok[(df_antrian_dok['Faskes'] == st.session_state.faskes_dokter) & (df_antrian_dok['Status_Periksa'] != 'Selesai')]
        
        if pasien_tunggu.empty:
            st.info("Bagus! Tidak ada antrian pasien saat ini.")
        else:
            pilih_pasien = st.selectbox("Daftar Pasien di Ruang Tunggu:", pasien_tunggu['Nama_Pasien'].tolist())
            data_pilih = pasien_tunggu[pasien_tunggu['Nama_Pasien'] == pilih_pasien].iloc[0]
            
            kel_mentah = data_pilih['Keluhan']
            keluhan_bersih = "-" if pd.isna(kel_mentah) or str(kel_mentah).lower() == 'nan' or str(kel_mentah) == "" else str(kel_mentah)
            
            st.markdown(f"### Riwayat Rekam Medis - {pilih_pasien}")
            df_rm_all = pd.read_excel(DB_REKAM_MEDIS)
            riwayat = df_rm_all[df_rm_all['Nama_Pasien'] == pilih_pasien]
            if riwayat.empty:
                st.caption("Ini adalah kunjungan pertama pasien (Tidak ada riwayat).")
            else:
                st.dataframe(riwayat[['Tanggal', 'Faskes', 'Diagnosis', 'Resep_Obat']], use_container_width=True, hide_index=True)

            st.markdown("### Pemeriksaan Saat Ini")
            
            # Bersihkan nilai 'nan' pada status rujukan
            ruj_mentah = data_pilih['Status_Rujukan']
            ruj_bersih = "" if pd.isna(ruj_mentah) or str(ruj_mentah).lower() == 'nan' or str(ruj_mentah) == "" else f" {ruj_mentah}"
            
            st.warning(f"**Keluhan:** {keluhan_bersih} | **Tipe:** {data_pilih['Jenis']}{ruj_bersih}")
            
            with st.form("form_pemeriksaan"):
                diagnosis = st.text_area("1. Catatan Rekam Medis / Diagnosis (Wajib)")
                resep = st.text_area("2. E-Resep / Peresepan Obat (Format bebas)")
                surat_sakit = st.number_input("3. Surat Keterangan Istirahat Sakit (Berapa Hari?)", min_value=0, max_value=30, value=0)
                surat_str = f"Istirahat {surat_sakit} Hari" if surat_sakit > 0 else "-"
                
                status_tindakan = st.radio("4. Tindakan Lanjutan", ["Selesai / Pulang", "Rujuk ke RS/Spesialis", "Rujuk Laboratorium Internal"])
                # RS Rujukan mengambil data dari variabel DAFTAR_FASKES
                rs_rujuk = st.selectbox("Pilih Faskes Rujukan (Bila Perlu)", ["-"] + DAFTAR_FASKES + ["Lab Darah", "Radiologi"])
                
                if st.form_submit_button("Simpan & Selesai"):
                    if not diagnosis:
                        st.error("Diagnosis medis wajib diisi.")
                    else:
                        tujuan = rs_rujuk if "Rujuk" in status_tindakan else "-"
                        df_rm = pd.read_excel(DB_REKAM_MEDIS)
                        new_rm = pd.DataFrame({
                            'Tanggal': [get_today_str()], 'Nama_Pasien': [pilih_pasien], 'Faskes': [st.session_state.faskes_dokter], 
                            'Diagnosis': [diagnosis], 'Resep_Obat': [resep], 'Surat_Sakit': [surat_str], 'Rujukan_Tujuan': [tujuan]
                        })
                        df_rm = pd.concat([df_rm, new_rm], ignore_index=True)
                        df_rm.to_excel(DB_REKAM_MEDIS, index=False)
                        
                        idx_update = df_antrian_dok.index[(df_antrian_dok['Nama_Pasien'] == pilih_pasien) & (df_antrian_dok['Faskes'] == st.session_state.faskes_dokter)].tolist()[0]
                        df_antrian_dok.at[idx_update, 'Status_Periksa'] = 'Selesai'
                        df_antrian_dok.to_excel(DB_ANTRIAN, index=False)
                        
                        st.success("Pemeriksaan Selesai! Data Rekam Medis & E-Resep telah diterbitkan.")
                        st.rerun()
