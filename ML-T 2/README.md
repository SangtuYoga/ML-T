# Laporan Proyek Machine Learning - Sang Putu Yoga Pramana

## Project Overview
[Anime](https://p2k.unkris.ac.id/ind/1-3073-2962/Anime_31456_s2-unkris_p2k-unkris.html) (アニメ) adalah animasi khas Jepang, yang biasanya dicirikan melewati gambar-gambar berwarna-warni yang mempertunjukkan tokoh-tokoh dalam bermacam macam lokasi dan cerita, yang ditujukan pada berbagai macam penonton. Anime dipengaruhi gaya gambar manga, komik khas Jepang. Kata anime tampil dalam bangun tulisan dalam tiga karakter katakana a, ni, me (アニメ) yang adalah bahasa serapan dari bahasa Inggris "Animation" dan dikatakan bagi "Anime-shon".

Seiring perkembangannya jaman, anime sudah menjadi hal yang wajar bagi para pecinta animasi terutama animasi jepang. Saat ini, ada banyak sekali platform yang memudahkan kita untuk menonton serial anime. Mulai dari layanan tv, hingga layanan streaming anime online. Dengan demikian kita sangat mudah untuk mengakses sebuah anime dan karena hal tersebut anime juga menjadi makin populer dikalangan remaja hingga dewasa contohnya seperti di Indonesia. Tetapi masih banyak orang yang memerlukan rekomendasi anime berdasarkan rating dari anime yang mereka tonton ataupun genre dari anime yang mereka telah tonton namun beberapa dari mereka tidak tau apakah anime tersebut sesuai dengan kriteria yang mereka inginkan atau tidak. Maka dari itu, diperlukannya Sistem rekomendasi untuk menentukan rekomendasi anime yang cocok dengan user tersebut. Dimana Sistem rekomendasi adalah suatu aplikasi untuk menyediakan dan merekomendasikan suatu item dalam membuat suatu keputusan yang diinginkan oleh pengguna (Ungkawa, et al., 2013).

## Business Understanding
### Problem Statement
Permasalahan inti dari projek ini adalah karena selalu banyak anime yang dirilis setiap seasonnya yakni tepatnya 6 bulan sekali, maka terdapat penggemar anime ragu untuk memilih anime yang ingin ditonton musim ini atau anime yang sudah lewat. Oleh karena itu, diperlukannya sistem rekomendasi dimana sistem tersebut bisa memberi anime yang tepat untuk user. 

### Goal
Tujuam dari Proyek ini adalah Memberikan rekomendasi judul anime kepada pengguna berdasarkan data genre dan memberikan rekomendasi anime kepada sebuah user bedasarkan hasil review pengguna lain terhadap anime.

### Solution
Berdasarkan dataset dari proyek ini karena hanya terdapat data mengenai rating dan detail anime seperti genre, saya akan menggunakan metode Content-Based Filtering dan Collaborative Filtering.
Dimana Content-Based Filtering berguna untuk merekomendasikan anime berdasarkan genre untuk modelnya saya menggunakan model cosine similatiry, Sedangkan Collaborative Filtering berguna untuk merekomendasikan anime kepada sebuah user berdasarkan penilaian dari seluruh pengguna/komunitas untuk modelnya saya menggunakan deep learning.

## Data Understanding
![banner](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/bannerdataset.png?raw=true)
Dataset ini didapat dari [kaggle](https://www.kaggle.com/).  Untuk projek ini, saya mengambil data yang bernama [Anime Recommendations Database vol.2](https://www.kaggle.com/noiruuuu/anime-recommendations-database-vol2). Berikut adalah keterangan mengenai maksud dari variabel - variabel atau kolom tersebut :

- animes.csv
    - anime_id - ID unik myanimelist.net yang mengidentifikasi anime.
    - judul - nama judul lengkap anime.
    - genre - daftar genre yang dipisahkan koma untuk anime ini.
    - media - film, TV, OVA, dll.
    - episode - berapa banyak episode dalam acara ini. (1 jika film atau ova).
    - rating - rating rata-rata dari 10 untuk anime ini.
    - anggota - jumlah anggota komunitas yang ada di "grup" anime ini.
    - start_date - kapan anime ini dimulai.
    - season - season berapa anime ini dimulai.
    - sumber - manga, light_novel, orisinal, dll.
- ratings.csv
    - user_id - ID pengguna yang dihasilkan secara acak tidak dapat diidentifikasi.
    - anime_id - anime yang telah dinilai oleh pengguna ini.
    - peringkat - peringkat dari 10 yang telah ditetapkan pengguna ini (0 jika pengguna menontonnya tetapi tidak memberikan peringkat).

Dalam proses data understanding, saya melakukan visualisasi data terhadap tipe dari media streaming dan genre yang paling sering terdapat/muncul dalam sebuah anime. Berikut adalah visualisasi dari data media streaming seluruh anime pada animes.csv:

![mediastreaming](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/mediastreaming.png?raw=true)

Dari hasil tersebut, dapat disimpulkan bahwa media streaming anime terbanyak terdapat pada tv dengan jumlah 3825 anime.

Selanjutnya saya melihat jenis genres yang paling banyak terdapat dalam data tersebut. Saya ingin melihat genre apa yang paling sering muncul dalam dataset ini. Untuk itu saya akan melakukan visualisasi data dengan menggunakan wordcloud.  Berikut adalah visualisasi dari data genre anime yang paling sering muncul pada animes.csv:

![genre](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/genres.png?raw=true)

Dari hasil tersebut, dapat disimpulkan bahwa dari seluruh genre yang ada. genre Comedy dan Action merupakan genre yang paling sering muncul atau sering terdapat dalam anime dalam dataset.

Selain itu saya juga memvisualisasikan data dalam bentuk tabel berdasarkan rating tertinggi dari seluruh media streaming dan rating tertinggi dari media streaming "movie". Karena saya ingin melihat anime yang terpopuler berdasarkan ketentuan diatas. Berikut merupakan visualisasi dari data tersebut.

**10 Anime dengan rating tertinggi**

![top10rating](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/top10rating.png?raw=true)

**10 Anime media streaming movie dengan rating tertinggi**

![top10movierating](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/top10movierating.png?raw=true)

## Data Preparation 

Dalam data preparation, ada beberapa teknik yang saya gunakan untuk proses *preparation*. Selain itu, ada 3 dataset yang saya akan periksa yaitu rating.csv yang dinamakan sebagai df_rating, Movie.csv yand dinamakan sebagai df_movie, dan gabungan kedua dataset yang dinamakan df. Berikut penjelasan beberapa teknik yang akan digunakan untuk *data preparation*dan hasil dari teknik tersebut :

1. Melakukan text cleaning terhadap judul anime
    Dikarenakan Beberapa judul anime menggunakan huruf jepang atau karakter khusus, maka dari itu dibuatkan fungsi untuk melakukan text cleaning. Berikut merupakan fungsi dalam melakukan text cleaning.
    ![textcleaning](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/textcleaning.png?raw=true)

2. Mengecek data apakah terdapat data null atau tidak
    Dengan adanya data null dapat membuat suatu hasil prediksi model menjadi tidak akurat. Berikut cara untuk melihat dan mengatasi hal tersebut.

    ![nullcleaning](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/nullcleaning.png?raw=true)

    Dapat kita lihat pada kolom genres memiliki 3 data yang kosong, Rating memiliki 3345 data yang kosong, dan members memiliki 16 data yang kosong, selain itu juga dapat melihat persentase dari data kosong tersebut. Sehingga, untuk mengatasi data null maka dilakukan pembersihan dengan menghapusnya.

3. Cosine Similatiry
    Menghitung cosine similairity dari setiap dataset menggunakan fungsi cosine_similarity dari library Sklearn. Pada tahapan ini, menghitung cosine similarity pada dataset dengan fungsi 'genre_recommendations' untuk pemberian rekomendasi anime berdasarkan kesamaan genre. 
    
    ![cosinesimilatiry](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/cosine.png?raw=true)    
    
4. Melakukan Label Encoder
    Untuk label yang akan digunakan hanya pada user_id dan anime_id yang hasilnya akan digunakan untuk melakukan model deep learning.
    
    ![labelencoder](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/labelencoder.png?raw=true)    


## Modeling and Result

### Cosine Similatiry
Berikut merupakan hasil dari modelling menggunakan *Cosine Simirality* untuk sistem rekomendasi berbasis *content-based filtering*. 

![content-based filtering](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/cbs.png?raw=true)    

Dari hasil diatas dapat disimpulkan bahwa terdapat 10 anime dengan kesamaan genre yang tinggi dengan anime "One Piece"

### Deep Learning
Selanjutnya merupakan hasil dari modelling *deep learning* untuk sistem rekomendasi berbasis *collaborative filtering*. Dimana model ini akan memberikan rekomendasi anime untuk seorang pengguna berdasarkan idnya. Disini saya membuat 2 model untuk collaborative filltering yang akan digunakan sebagai pembanding model manakah yang terbaik untuk melakukan rekomendasi terhadap seorang user.
- Hasil Model 1

![collaborative filtering 1](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/model1.png?raw=true)    

- Hasil Model 2

![collaborative filtering 2](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/model2.png?raw=true)   
    
Dari kedua gambar tersebut dapat disimpulkan bahwa model pertama adalah model yang terbaik dikarenakan rating prediksinya terhadap sebuah anime lebih tinggi dibandingkan model kedua. 

## Evaluation 

Dalam proses evaluasi, akan disajikan informasi mengenai perbandingan mengenai model pertama dan kedua melalui dua metrik berikut:

### Loss (Mean Squared Error Loss)

Mean Squared Erorr Loss berfungsi untuk menghitung rata-rata kuadrat kesalahan antara label dan prediksi. Dengan demikian semakin rendahnya nilai loss (mean squared error loss) maka semakin baik dan akurat model yang dibuat. Berikut adalah hasil perbandingan loss dan val_loss pada kedua model yang telah dibuat.

![loss](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/metrikloss.png?raw=true) 
![valloss](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/metrikvalloss.png?raw=true) 

### RMSE (Root Mean Squared Error)

Root Mean Squared Error adalah matrik yang berfungsi untuk menghitung kuadrat dari rata-rata selisih kuadrat antara nilai taksiran dan nilai sebenarnya dari variabel/fitur. Dengan demikian semakin rendahnya nilai RMSE makan semakin baik model tersebut dalam melakukan prediksi. Berikut formula/rumus dari root mean squared error.

![formularmse](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/rmse.png?raw=true) 

Berikut adalah hasil perbandingan root_mean_squared_error dan val_root_mean_squared_error pada kedua model yang telah dibuat.

![loss](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/metrikrmse.png?raw=true) 
![loss](https://github.com/SangtuYoga/ML-T/blob/main/ML-T%202/metrikvalrmse.png?raw=true) 

Dapat disimpulkan bahwa dari pembuatan sistem rekomendasi dengan mengunakan Content-Based Recommendation system (Cosine Similatiry) dan Collaborative Filtering(Deep Learning) yakni mendapatkan hasil yang sesuai seperti yang saya harapkan, yaitu dapat merekomendasikan anime berdasarkan genre yang sejenis dengan kesamaan yang cukup tinggi dan dapat merekomendasikan anime terhadap sebuah user dengan prediksi yang cukup baik.
