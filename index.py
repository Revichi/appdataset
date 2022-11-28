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
    Dataset Brain Tumor otak yang mencakup lima fitur ukuran pertama dan delapan fitur tekstur dengan level target.
    """)
    st.write("""
        Fitur Peratama: Mean, Variance,Standard Deviation,Skewness,Kurtosis
    """)
    st.write("""Fitur Kedua :
        ,Contrast
        ,Energy
        ,ASM (Angular second moment)
        ,Entropy
        ,Homogeneity
        ,Dissimilarity
        ,Correlation
        ,Coarseness""")
    data = pd.read_csv("DataMin/BrainTumor.csv")
    data.fillna(0,inplace=True)
    data

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
                
                
with st.sidebar:
    st.write('Link:')
    colum = st.columns((0.1,1,1))
    url = 'https://github.com/Revichi/appdataset'
    if colum[1].button('GitHub'):
        webbrowser.open_new_tab(url)      
    link = 'https://revichi.github.io/datamining/App.html?highlight=penambangan'
    if colum[2].button('Jupyter'):
        webbrowser.open_new_tab(link)
