import streamlit as st
import numpy as np
import aksi
import time
import webbrowser
import pandas as pd

with st.sidebar:
    kolom = st.columns((1, 1, 2.7))
    home = kolom[1].button('Home',type='primary')
    Tools = kolom[2].button('Tools')

if Tools==False and home==True:
    st.title('Brain Tumor')
    st.write("""
    Dataset Brain Tumor otak yang mencakup 13 fitur yaitu Lima fitur ukuran dan delapan fitur tekstur dengan level target.
    """)

    st.header (" Fitur Ukuran:") 
    st.subheader("Mean ")
    st.write("Nilai rata-rata yang merupakan hasil dari pembagian antara jumlah seluruh nilai data dengan jumlah data yang ada.")
    st.subheader("Variance")
    st.write("Suatu ukuran penyebaran data, yang diukur dalam pangkat dua dari selisih data terhadap rata-ratanya.")
    st.subheader("Standard Deviation")
    st.write("Merupakan suatu nilai yang digunakan dalam menentukan persebaran data pada suatu sampel dan melihat seberapa dekat data-data tersebut dengan nilai mean. Standar deviasi atau simpangan baku merupakan ukuran penyebaran yang paling baik, karena menggambarkan besarnya penyebaran tiap-tiap unit observasi")
    st.subheader("Skewness")
    st.write("Merupakan Kemiringan (ketidaksimetrian) suatu data statistik atau dapat pula didefinisikan sebagai penyimpangan dari kesimetrian suatu distribusi")
    st.subheader("Kurtosis")
    st.write("Keruncingan distribusi data  atau ukuran tinggi rendahnya puncak suatu distribusi data terhadap distribusi normalnya data.")
    
    st.header("Fitur Tekstur:")
    st.subheader("Contrast")
    st.write("Suatu ukuran intensitas aras keabuan antara piksel dengan piksel lainnya")
    st.subheader("Energy")
    st.write("Ukuran yang menyatakan distribusi intensitas piksel terhadap jangkauan aras keabuan.")
    st.subheader("ASM (Angular second moment)")
    st.write("Ukuran homogenitas citra atau keseragaman pada citra.")
    st.subheader("Entropy")
    st.write("Ukuran ketidakaturan aras keabuan dalam suatu citra")
    st.subheader("Homogeneity")
    st.write("Merupakan ukuran yang digunakan untuk mengukur kehomogenan variasi intensitas dalam citra ")
    st.subheader("Dissimilarity")
    st.write("Merupakan ketidakmiripan pada suatu tekstur/Mengukur ketidakmiripan suatu tekstur, yang akan bernilai besar bila acak dan sebaliknya akan bernilai kecil bila seragam.")
    st.subheader("Correlation")
    st.write("Korelasi menunjukkan ketergantungan linier derajat keabuan dari pikselpiksel yang saling bertetangga dalam suatu citra abu-abu.")
    st.subheader("Coarseness")
    st.write("Merupakan fitur tekstur yang paling fundamental pada pengelolaan citra. Semakin besar jarak dari elemen sebuah citra maka citra tersebut semakin kasar. Coarseness juga disebut Kekasaran(Kekasaran Pada Citra)")
    st.header("Dataset :")
    data = pd.read_csv('DataMin/BrainTumor.csv')
    data.fillna(0,inplace=True)
    data

if home==False and Tools==False or home==False and Tools==True:
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

    st.write("""Metode Klasfikasi""")
    model_1 = st.checkbox('K-Nearest Neighbors', value=True)
    model_2 = st.checkbox('Naive Bayes ')
    model_3 = st.checkbox('Decision Tree')

    columns = st.columns((2, 0.6, 2))
    sumbit = columns[1].button("Submit")
    if sumbit:
        # normalisasi data
        data = aksi.normalisasi([Mean,Variance,StandardDeviation,Entropy,Skewness,Kurtosis,Contrast,Energy,ASM,Homogeneity,Dissimilarity,Correlation,Coarseness])
        # prediksi data
        if model_1 or model_2 or model_3:
            if model_1:
                prediksi = aksi.knn(data)
                # cek prediksi
                with st.spinner("Tunggu Sebentar Masih Proses..."):
                    if prediksi[-1] == 0:
                        time.sleep(1)
                        st.success("Hasil Prediksi Metode KNN: "+nama+" Tidak Memiliki Tumor")
                    else:
                        time.sleep(1)
                        st.warning("Hasil Prediksi Metode KNN: "+nama+" Memiliki Tumor")

            if model_2:
                prediksi = aksi.nb(data)
                # cek prediksi
                with st.spinner("Tunggu Sebentar Masih Proses..."):
                    if prediksi[-1] == 0:
                        time.sleep(1)
                        st.success("Hasil Prediksi Metode Naive Bayes: "+nama+" Tidak Memiliki Tumor")
                    else:
                        time.sleep(1)
                        st.warning("Hasil Prediksi Metode Naive Bayes: "+nama+" Memiliki Tumor")

            if model_3:
                prediksi = aksi.nb(data)
                # cek prediksi
                with st.spinner("Tunggu Sebentar Masih Proses..."):
                    if prediksi[-1] == 0:
                        time.sleep(1)
                        st.success("Hasil Prediksi Metode Decision Tree: "+nama+" Tidak Memiliki Tumor")
                    else:
                        time.sleep(1)
                        st.warning("Hasil Prediksi Metode Decision Tree: "+nama+" Memiliki Tumor")
        else:
            st.error("Pilih Salah Satu Metode")


with st.sidebar:
    st.write('Link:')
    link = '[GitHub](https://github.com/Revichi/appdataset)'
    st.markdown(link, unsafe_allow_html=True)
    link = '[Jupyter Book](https://revichi.github.io/datamining/App.html?highlight=penambangan)'
    st.markdown(link, unsafe_allow_html=True)
