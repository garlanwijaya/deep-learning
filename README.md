# Laporan Benchmark: Plain-34 vs. ResNet-34

Laporan ini membandingkan performa arsitektur Plain-34 (tanpa *residual connection*) dengan arsitektur ResNet-34 pada dataset 5 Makanan Indonesia. Eksperimen ini bertujuan untuk menunjukkan dampak dari *residual connection* dalam mengatasi masalah degradasi pada jaringan yang dalam.

---

## 1. Konfigurasi Eksperimen

Berikut adalah hyperparameter yang digunakan secara konsisten untuk kedua model demi perbandingan yang adil.

- **Arsitektur Dasar:** ResNet-34 (3, 4, 6, 3 blok)
- **Learning Rate:** 0.001
- **Optimizer:** Adam
- **Loss Function:** Cross Entropy Loss
- **Jumlah Epoch:** 5
- **Batch Size:** 32

---

## 2. Hasil Performa

Tabel berikut merangkum metrik performa kedua model pada **epoch terakhir** dari proses pelatihan.

| Model     | Validation Accuracy | Validation Loss |
|-----------|---------------------|-----------------|
| Plain-34  | **0.5405** | **1.1435** |
| ResNet-34 | **0.6081** | **0.9751** |

> **Tips:** Untuk mendapatkan nilai di atas, jalankan kode ini di sel baru Colab:
> ```python
> print(f"Plain-34  -> Val Acc: {plain_history['val_acc'][-1]*100:.2f}%, Val Loss: {plain_history['val_loss'][-1]:.4f}")
> print(f"ResNet-34 -> Val Acc: {resnet_history['val_acc'][-1]*100:.2f}%, Val Loss: {resnet_history['val_loss'][-1]:.4f}")
> ```

---

## 3. Kurva Pelatihan

Grafik di bawah ini memvisualisasikan perbandingan performa akurasi dan loss dari kedua model selama proses pelatihan.
<img width="1489" height="590" alt="grafik" src="https://github.com/user-attachments/assets/e8113072-a70e-48ac-946b-1ffa6805f08a" />


---

## 4. Analisis Singkat

Berdasarkan hasil eksperimen yang divisualisasikan pada grafik, terlihat jelas bahwa **penambahan *residual connection* pada arsitektur ResNet-34 memberikan peningkatan performa yang signifikan**.

Model **Plain-34** menunjukkan gejala **degradasi**. Meskipun *training accuracy*-nya terus meningkat, *validation accuracy*-nya cenderung stagnan di angka **[lihat angka stagnasi di grafik]%** setelah epoch ke-**[lihat epoch ke berapa]**. Hal ini membuktikan bahwa model kesulitan untuk belajar lebih dalam dan menggeneralisasi pengetahuannya ke data yang belum pernah dilihat.

Sebaliknya, model **ResNet-34** berhasil mengatasi masalah ini. Kurva *validation accuracy*-nya menunjukkan tren peningkatan yang lebih konsisten dan mencapai puncak yang lebih tinggi. *Residual connection* memungkinkan gradien untuk mengalir lebih mudah ke lapisan-lapisan awal, sehingga model dapat terus belajar secara efektif tanpa mengorbankan performa. Peningkatan performa ini membuktikan efektivitas arsitektur ResNet dalam melatih jaringan yang dalam.
