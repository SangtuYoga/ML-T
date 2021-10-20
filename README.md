# Laporan Proyek Machine Learning - Sang Putu Yoga Pramana

## Domain Proyek
Kesehatan adalah hal yang sangat penting untuk diperhatikan bagi semua manusia karena jika tubuh sehat maka kita dapat melakukan aktivitas sehari- hari dengan baik. Namun, tidak ada yang dapat memastikan bahwa seseorang akan sehat selamanya. Resiko sakit bisa terjadi kapanpun dan kepada siapapun. Jika kita sakit salah satu yang menjadi masalah utama yaitu mengenai biaya kesehatan yang tidak murah dan kurangnya persiapan dana mengenai hal tersebut. Oleh karena itu, sangatlah diperlukan suatu persiapan untuk mengatasi risiko sakit tersebut salah satunya dengan berpartisipasi dalam asuransi kesehatan. Asuransi kesehatan adalah asuransi yang memberikan jaminan kepada tertanggung untuk mengganti setiap biaya pengobatan meliputi biaya perawatan rumah sakit, biaya pembedahan dan biaya obat- obatan. Pada website [Qoala](https://www.qoala.app/id/blog/asuransi/kesehatan/alasan-pentingnya-asuransi-kesehatan/) memberikan alasan betapa pentingnya asuransi kesehatan untuk dimiliki. Pada proyek ini saya memilih topik untuk memprediksi premi asuransi biaya pengobatan di masa depan. Premi ini ialah iuran biaya yang harus dibayarkan oleh nasabah selama jangka waktu yang sudah disepakati. Data dalam proyek ini berisi mengenai data nasabah seperti age, sex, bmi, children, smoker, region, expenses. Untuk menjawab masalah ini, predictive analytics diharapkan dapat memprediksi masalah tersebut dan mendapatkan solusi yang terbaik dengan menggunakan model machine learning. 

## Business Understanding
### Problem Statement
Berikut adalah problem statement dari proyek ini:
* Apakah ada hal yang mempengaruhi asuransi perjalanan seseorang individu?</br>
* Manakah Model Machine Learning yang terbaik dalam menyelesaikan permasalahan ini?</br>
### Goals
Berikut adalah goals yang ingin dicapai dalam proyek ini:
*	Mengetahui hal yang mempengaruhi asuransi perjalanan seseorang individu
*	Mengetahui model terbaik dalam Machine Learning untuk memprediksi asuransi perjalanan seseorang.
### Solution Statements
Solusi yang diajukan antara lain adalah Decision Tree, Support Vector Machine dan Random Forest. Dengan pengertian:
*	**Decision Trees (DTs**
</br>[Decision Trees (DTs)](https://scikit-learn.org/stable/modules/tree.html) adalah metode pembelajaran terawasi non-parametrik yang digunakan untuk klasifikasi dan regresi. Tujuannya adalah untuk membuat model yang memprediksi nilai variabel target dengan mempelajari aturan keputusan sederhana yang disimpulkan dari fitur data. Sebuah pohon dapat dilihat sebagai pendekatan konstan sepotong demi sepotong.
*	**Support Vector Machine**
</br>[Support Vector Machine](https://scikit-learn.org/stable/modules/svm.html) adalah seperangkat metode pembelajaran terawasi yang digunakan untuk klasifikasi, regresi, dan deteksi outlier .
*	**Random Forest**
</br>[Regresi Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) adalah meta estimator yang cocok dengan sejumlah pengklasifikasi pohon keputusan pada berbagai sub-sampel dari dataset dan menggunakan rata-rata untuk meningkatkan akurasi prediksi dan kontrol over-fitting. Ukuran sub-sampel dikontrol dengan parameter max_samples jika bootstrap=True (default), jika tidak, seluruh dataset digunakan untuk membangun setiap pohon.
## Data Understanding
Dataset yang digunakan adalah dataset [Travel Insurance Prediction Data](https://www.kaggle.com/tejashvi14/travel-insurance-prediction-data) dari situs Kaggle yang berisi data tentang Asuransi Ditawarkan Kepada Beberapa Pelanggan Pada Tahun 2019 Dan Data Yang Diberikan Telah Disarikan Dari Kinerja/Penjualan Paket Selama Periode Itu dengan Age, Employment Type, GraduateOrNot, AnnualIncome, FamilyMembers, ChronicDisease, FrequentFlyer, EverTravelledAbroad dan TravelInsurance yang menjadi label pada data ini. Dataset ini memiliki 1987 dengan 9 kolom dengan 4 kategorikal dan 5 numerikal sebagai berikut :
* Age - Usia dari pelanggan
* Employment Type - Sektor bagian mana pelanggan berkerja
* GraduateOrNot -  Apakah pelanggan lulusan perguruan tinggi atau tidak
    * yes : lulusan perguruan tinggi
    * no : tidak lulusan perguruan tinggi
* AnnualIncome - Pendapatan Tahunan Pelanggan Dalam Rupee India[Dibulatkan ke Terdekat 50 Ribu Rupee]
* FamilyMembers - Jumlah Anggota Keluarga Pelanggan
* ChronicDisease - Apakah Pelanggan Menderita Penyakit Atau Kondisi Utama Seperti Diabetes
    * 1 : menderita penyakit
    * 0 : tidak menderita penyakit
* FrequentFlyer - Apakah pelanggan pernah mendapatkan program penumpang setia atau tidak
    * yes : pernah
    * no : tidak pernah
* EverTravelledAbroad - Apakah Pelanggan Pernah Bepergian Ke Luar Negeri atau tidak
    * yes : pernah
    * no : tidak pernah
* TravelInsurance - Apakah Pelanggan pernah Membeli Paket Asuransi Perjalanan
    * 1 : pernah
    * 0 : tidak pernah

Apabila dilakukan Data Loading adalah sebagai berikut.
|id|Age|Employment Type|GraduateOrNot|AnnualIncome|FamilyMembers|ChronicDiseases|FrequentFlyer|EverTravelledAbroad|TravelInsurance|
|--|:--:|----|-----|:---:|-------:|:--------:|-----------:|-----------:|-----------:|
|0|31|Government Sector|Yes|400000|6|1|No|No|0|
|1|31|Private Sector/Self Employed|Yes|1250000|7|0|No|No|0|
|2|34|Private Sector/Self Employed|Yes|500000|4|1|No|No|1|
|3|28|Private Sector/Self Employed|Yes|700000|3|1|No|No|0|
|4|28|Private Sector/Self Employed|Yes|700000|8|1|Yes|No|0|
|...|...|...|...|...|...|...|...|...|...|
|1982|33|Private Sector/Self Employed|Yes|1500000|4|0|Yes|Yes|1|
|1983|28|Private Sector/Self Employed|Yes|1750000|5|1|No|Yes|0|
|1984|28|Private Sector/Self Employed|Yes|1150000|6|1|No|No|0|
|1985|34|Private Sector/Self Employed|Yes|1000000|6|0|Yes|Yes|1|
|1986|34|Private Sector/Self Employed|Yes|500000|4|0|No|No|0|

Dataset tersebut juga dapat dilihat deskripsi statistiknya seperti berikut:

|Jenis|year|AnnualIncome|FamilyMembers|ChronicDiseases|TravelInsurance|
|---|---|---|---|---|---|
|count|1987.000000|1.987000e+03|1987.000000|1987.000000|1987.000000|
|mean|29.650226|9.327630e+05|4.752894|0.277806|0.357323|
|std|2.913308|3.768557e+05|1.609650|0.448030|0.479332|
|min|25.000000|3.000000e+05|2.000000|0.000000|0.000000|
|25%|28.000000|6.000000e+05|4.000000|0.000000|0.000000|
|50%|29.000000|9.000000e+05|5.000000|0.000000|0.000000|
|75%|32.000000|1.250000e+06|6.000000|1.000000|1.000000|
|max|35.000000|1.800000e+06|9.000000|1.000000|1.000000|

**Visualisasi Data**
</br>Mengecek outlier pada data dengan menggunakan boxplot.

![Boxplot1](https://github.com/SangtuYoga/ML-T/blob/main/boxplot1.png?raw=true)
![Boxplot2](https://github.com/SangtuYoga/ML-T/blob/main/boxplot2.png?raw=true)
![Boxplot3](https://github.com/SangtuYoga/ML-T/blob/main/boxplot3.png?raw=true)
![Boxplot4](https://github.com/SangtuYoga/ML-T/blob/main/boxplot4.png?raw=true)
![Boxplot5](https://github.com/SangtuYoga/ML-T/blob/main/boxplot5.png?raw=true)
</br>Karena tidak terdapat outlier jadi kita tidak perlu melakukan IQR method.

**Categorical Feature:**
</br>Disini saya menganalisi fitur kategorikal dengan menggunakan countplot

![Countplot1](https://github.com/SangtuYoga/ML-T/blob/main/cp0.png?raw=true)
![Countplot1](https://github.com/SangtuYoga/ML-T/blob/main/cp1.png?raw=true)
![Countplot1](https://github.com/SangtuYoga/ML-T/blob/main/cp2.png?raw=true)
![Countplot1](https://github.com/SangtuYoga/ML-T/blob/main/cp3.png?raw=true)

Dilihat dari diagram diatas dapat disimpulkan bahwa :
* Pada diagram Emploment Type terdapat sekitar 70% masuk ke dalam tipe Private Sector/Self Employed dan 30% ke dalam tipe Goverment Sector.
* Pada diagram GraduateOrNot terdapat sekitar 85% yang lulusan perguruan tinggi dan 15% tidak lulusan perguruan tinggi.
* Pada diagram Program penumpang setia(Frequent Flyer) terdapat sekitar 80% tidak pernah mendapatkan program tersebut dan 20% pernah mendapatkan program tersebut.
*  Pada diagram EverTravelledAbroad terdapat sekitar 80% tidak pernah berpergian ke luar negeri dan 20% pernah berpergian ke luar negeri.

**Numerical Feature:**
</br>Disini saya menganalisi fitur kategorikal dengan menggunakan histogram

![hist](https://github.com/SangtuYoga/ML-T/blob/main/hist.png?raw=true)

Dari hasil diatas dapat disimpulkan bahwa :
* Sebagian besar individu berusia 28 tahun.
* Sebagian besar individu (lebih dari 140) memiliki pendapatan tahunan terbanyak
* Ukuran keluarga maksimum adalah sekitar 4 anggota.
* Sebagian besar individu tidak menderita penyakit
* Sebagian besar individu tidak pernah mendapatkan asuransi perjalanan

Visualisasi menggunakan pairplots dan heatmap.
</br>Menggunakan parameter hue yakni EverTravelledAbroad yang berfungsi mengelompokkan variabel yang akan menghasilkan data point dengan warna berbeda sesuai kategorinya.

![heatmap](https://github.com/SangtuYoga/ML-T/blob/main/heatmap.png?raw=true)
![pairplot](https://github.com/SangtuYoga/ML-T/blob/main/pairplot.png?raw=true)

Dari hasil diatas dapat disimpulkan bahwa :
* Sebaigian besar umur yang diatas 26 tahun belum pernah berpergian keluar negeri.
* Sebagian keluarga yang memiliki anggota keluarga dibawah 4 pernah berpergian keluar negeri.
* Tidak semua yang  pernah berpergian keluar negeri mendapatkan Asuransi perjalanan.
* Banyak yang tidak berpergian keluar negeri juga tidak mendapatkan asuransi perjalanan.

## Data Preparation
Pada data preparation ini saya menggunakan One-Hot Encoding. Dimana One-Hot encoding adalah salah satu metode encoding. Metode ini merepresentasikan data bertipe kategori sebagai vektor biner yang bernilai integer, 0 dan 1, dimana semua elemen akan bernilai 0 kecuali satu elemen yang bernilai 1, yaitu elemen yang memiliki nilai kategori tersebut. Berikut hasil implementasi One- Hot-Encoding pada proyek saya.

|id|Age|Employment Type|GraduateOrNot|AnnualIncome|FamilyMembers|ChronicDiseases|FrequentFlyer|EverTravelledAbroad|TravelInsurance|
|--|:--:|----|-----|:---:|-------:|:--------:|-----------:|-----------:|-----------:|
|0|31|Government Sector|Yes|400000|6|1|No|No|0|
|1|31|Private Sector/Self Employed|Yes|1250000|7|0|No|No|0|
|2|34|Private Sector/Self Employed|Yes|500000|4|1|No|No|1|
|3|28|Private Sector/Self Employed|Yes|700000|3|1|No|No|0|
|4|28|Private Sector/Self Employed|Yes|700000|8|1|Yes|No|0|

**Menjadi :**

|id|Age|AnnualIncome|FamilyMembers|ChronicDiseases|TravelInsurance|Employment Type_Private Sector/Self Employed|GraduateOrNot_Yes|FrequentFlyer_Yes|EverTravelledAbroad_Yes|
|--|:--:|----|-----|:---:|-------:|:--------:|-----------:|-----------:|-----------:|
|0|31|400000|6|1|0|0|1|0|0|
|1|31|1250000|7|0|0|1|1|0|0|
|2|34|500000|4|1|1|1|1|0|0|
|3|28|700000|3|1|0|1|1|0|0|
|4|28|700000|8|1|0|1|1|1|0|

Kemudian melakukan proses Train-Test Split. Dimana proses ini adalah pembagian dataset menjadi data latih (train) dan data uji (test) merupakan hal yang saya pilih untuk lakukan sebelum membuat model. Hal ini karena data uji berperan sebagai data baru yang benar-benar belum pernah dilihat oleh model sebelumnya sehingga informasi yang terdapat pada data uji tidak mengotori informasi yang terdapat pada data latih, alasan lain mengapa menggunakan train test split karena untuk efisiensi dan tidak melakukan data leakage ketika melakukan scaling.

```python
X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.8,random_state=42)
```
## Modeling
Pada Proyek yang dibuat, digunakan model Machine Learning yaitu Decision Tree, Support Vector Machine, dan Random Forest. Model tersebut digunakan karena permasalahan dari model Machine Learning yang saya buat adalah permasalahan regresi. Pada tahap ini saya juga melakukan improvement terhadap model dengan menggunakan hyperparameter tuning. Pada Decision Tree menggunakan parameter max_depth=2, random_state=5, pada Supprot Vector Machine menggunakan parameter kernel = 'poly',degree=2, dan pada Random Forest menggunakan parameter n_estimators=50, max_depth=2, n_jobs = 1, random_state=5. Lalu untuk membandingkan ketiga model yang saya gunakan ini dilakukan perhitungan dari nilai Score, R-Squared, Accuracy dan MSE dari data. Setelah dilakukan pelatihan maka dapat disimpulkan bahwa jika menggunakan model Decision Tree akan menghasilkan nilai accuracy yang tinggi yaitu 80% dan MSE yang rendah. 

## Evaluation
Pada tahap evaluation akan dijelaskan mengenai metrik yang digunakan dalam prediksi proyek saya dengan menggunakan metrik accuracy dan MSE. Dimana Akurasi merupakan metrik untuk menghitung nilai ketepatan model dalam memprediksi data dengan data yang sebenarnya dan MSE merupakan metrik untuk menghitung seberapa berbedanya seperangkat nilai. Semakin kecil nilai MSE, semakin dekat nilai yang diprediksi dan diamati. Untuk menghitung nilai dari Akurasi dan MSE menggunakan rumus berikut:

![akurasiformula](https://github.com/SangtuYoga/ML-T/blob/main/akurasi.png?raw=true)</br>
![mseformula](https://github.com/SangtuYoga/ML-T/blob/main/mse.png?raw=true)

Keterangan: 
* MSE = Nilai mean square error
* n = jumlah data sampel
* i = urutan data
* Y = Nilai hasil observasi
* Ŷ = Nilai hasil prediksi

Hasil dari evaluation model pada proyek ini mengenai prediksi biaya asuransi dapat dilihat pada gambar di bawah ini. 

![matrikmse](https://github.com/SangtuYoga/ML-T/blob/main/msematrik.png?raw=true)
![akurasimodel](https://github.com/SangtuYoga/ML-T/blob/main/akurasimodel.png?raw=true)


Jadi dapat disimpulkan bahwa model yang memiliki akurasi tertinggi dan MSE yang rendah dalam memprediksi asuransi perjalanan adalah menggunakan model Decision Tree Classifier dengan akurasi tertinggi yakni 80% dengan mse yang paling rendah diantara model lainnya.
