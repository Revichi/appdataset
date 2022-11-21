import streamlit as st
import numpy as np
import aksi
import time
import webbrowser

with st.sidebar:
    kolom = st.columns((1, 1, 2.7))
    home = kolom[1].button('Home',type='primary')
    Tools = kolom[2].button('Tools')

if home==True and Tools==False:
    st.title('Brain Tumor')
    st.write("""
    Dapat Memprediksi Tumor dengan menginputkan analisa hasil rontgen kepala
    """)

if Tools==False and home==False or Tools==True and home==False:
    st.title('Tools')
    st.write("""
    Harap Isi Data Sesuai Kolom, Data Tidak Boleh Kosong
    """)
    nama = st.text_input("Nama", placeholder= 'Nama')
    Mean = st.number_input("Mean")
    Variance = st.number_input("Variance")
    StandardDeviation = st.number_input("Standard Deviation")
    Entropy = st.number_input("Entropy")
    Skewness = st.number_input("Skewness")
    Kurtosis = st.number_input("Kurtosis")
    Contrast = st.number_input("Contrast")
    Energy = st.number_input("Energy")
    ASM = st.number_input("ASM(Angular Second Moment)")
    Homogeneity = st.number_input("Homogeneity")
    Dissimilarity = st.number_input("Dissimilarity")
    Correlation = st.number_input("Correlation")
    Coarseness = st.number_input("Coarseness")
    columns = st.columns((2, 0.6, 2))
    sumbit = columns[1].button("Submit")
    if sumbit:
        # normalisasi data
        data = aksi.normalisasi([Mean,Variance,StandardDeviation,Entropy,Skewness,Kurtosis,Contrast,Energy,ASM,Homogeneity,Dissimilarity,Correlation,Coarseness])
        # prediksi data
        prediksi = aksi.knn(data)    
        # cek prediksi
        with st.spinner("Tunggu Sebentar Masih Proses..."):
            if prediksi[-1]== 0:
                time.sleep(2)
                st.success("Hasil : "+nama+" Bukan Tumor")
            else :  
                time.sleep(2)
                st.warning("Hasil : "+nama+" Kemungkinan adalah tumor")