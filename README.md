# Credit Card Fraud Detection Dashboard

Aplikasi machine learning untuk deteksi fraud kartu kredit menggunakan pendekatan unsupervised learning dan clustering analysis.

## Live Demo

**Dashboard tersedia di:** [https://capstoneproject-01-glenferdinza.streamlit.app/](https://capstoneproject-01-glenferdinza.streamlit.app/)

> **Status**: Deployment sedang dalam proses. Jika terjadi error, akan diperbaiki secara otomatis.

## Deskripsi Project

Dashboard ini menggunakan pendekatan unsupervised learning untuk mengidentifikasi pola fraud dalam transaksi kartu kredit. Model dibangun menggunakan:

- **K-Means Clustering** untuk segmentasi customer
- **Feature Engineering** dengan 8 rasio keuangan teroptimasi
- **RandomForest Classifier** untuk prediksi fraud
- **Cross-validation** untuk validasi model

## Teknologi yang Digunakan

- **Python 3.8+**
- **Streamlit** - Web framework
- **Scikit-learn** - Machine learning
- **Pandas & NumPy** - Data processing
- **Matplotlib & Seaborn** - Visualisasi
- **XGBoost** - Advanced ML algorithms

## Features Dashboard

1. **Overview** - Insights dataset dan findings
2. **Feature Analysis** - Analisis feature engineering
3. **Fraud Detection** - Interface prediksi real-time
4. **Cluster Analysis** - Segmentasi customer
5. **Model Performance** - Perbandingan algoritma
6. **Validation Analysis** - Assessment metodologi
7. **Anomaly Detection** - Isolation Forest analysis

## Metodologi

### Data Processing
- Dataset: 8,950 customer kartu kredit
- Feature engineering: 8 rasio keuangan optimized
- Clustering: K-means untuk identifikasi pola

### Model Development
- **Unsupervised Learning**: Clustering untuk label generation
- **Supervised Learning**: Multiple algorithms comparison
- **Validation**: Cross-validation dan statistical testing

### Performance Metrics
- **Cross-validation Accuracy**: 99.81% ± 0.14%
- **Fraud Detection Recall**: 99.30%
- **F1-Score**: 99.39% ± 0.44%

## Important Notes

Model ini menggunakan **pseudo-labeling** approach dimana fraud labels dihasilkan dari clustering analysis, bukan dari ground truth data. Performance metrics mencerminkan konsistensi clustering, bukan validasi terhadap actual fraud cases.

## Deployment Instructions

### Local Development
```bash
# Clone repository
git clone https://github.com/Glenferdinza/CapstoneProject

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```
## File Structure

```
python_format/
├── app.py                           # Main dashboard
├── fraud_detection_pipeline.py     # Training pipeline
├── requirements.txt                 # Dependencies
├── optimized_clustering_cc.h5       # K-means model
├── optimized_scaler_cc.h5          # Data scaler
├── optimized_PCA_clustering_cc.h5  # PCA model
├── fraud_classifier_best.h5        # Best classifier
├── optimized_data_clustering_cc.csv # Processed data
└── CC GENERAL.csv                  # Original dataset
```

## Model Files Required

Pastikan semua file model berikut ada di repository:
- `optimized_clustering_cc.h5` - K-means clustering model
- `optimized_scaler_cc.h5` - Data preprocessing scaler
- `optimized_PCA_clustering_cc.h5` - PCA transformation model
- `fraud_classifier_best.h5` - Trained fraud classifier
- `optimized_data_clustering_cc.csv` - Processed dataset

## Performance Summary

| Metric | Value | Note |
|--------|--------|------|
| Accuracy | 99.81% | Cross-validation |
| Precision | 99.48% | Weighted average |
| Recall | 99.30% | Fraud detection |
| F1-Score | 99.39% | Balanced metric |

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/newFeature`)
3. Commit changes (`git commit -m 'Add newFeature'`)
4. Push to branch (`git push origin feature/newFeature`)
5. Open Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

**GlenFerdinza** - [GitHub](https://github.com/Glenferdinza)

Project Link: [https://github.com/Glenferdinza/CapstoneProject/tree/main](https://github.com/Glenferdinza/CapstoneProject/tree/main)
