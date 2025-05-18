import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from tqdm import tqdm
import concurrent.futures
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('vader_lexicon')

# Inisialisasi VADER SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Konfigurasi file
input_file = 'Dataset_ulasan_Erigo.csv'
output_file = 'Dataset_ulasan_Erigo_sentiment.csv'
mongodb_ready_file = 'ulasan_sentimen_mongodb.csv'  # File khusus untuk diimpor ke MongoDB
output_dir = 'output_sentiment'

# Buat direktori output jika belum ada
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Tambahkan kata positif dalam Bahasa Indonesia
pos_words = {
    'bagus': 2.0, 'pas': 1.5, 'sesuai': 2.0, 'pengiriman': 1.0, 'nyaman': 2.0, 'cepat': 1.5, 'banget': 2.0, 
    'oke': 1.0, 'baik': 1.0, 'sampai': 1.0, 'adem': 2.0, 'mantap': 2.0, 'aman': 1.5, 'sangat': 1.0, 
    'ok': 1.0, 'lumayan': 1.5, 'keren': 2.0, 'rapi': 2.0, 'packing': 1.0, 'tebal': 2.0, 'cukup': 1.5, 
    'halus': 2.0, 'cocok': 2.0, 'puas': 2.0, 'berkualitas': 2.0, 'langganan': 1.5, 'suka': 1.5, 
    'diterima': 1.0, 'rapih': 2.0, 'original': 2.0, 'lembut': 2.0, 'enak': 1.5, 'ori': 2.0, 'joss': 2.0, 
    'mantab': 2.0, 'tepat': 1.5, 'datang': 1.0, 'selamat': 1.0, 'terbaik': 2.0, 'terjamin': 1.0, 'cakep': 2.0, 
    'bersahabat': 1.0, 'murah': 1.5, 'okelah': 1.0, 'good': 1.0, 'recommended': 2.0, 'memuaskan': 2.0, 
    'ganteng': 2.0, 'baguss': 1.0, 'bener': 1.0, 'gercep': 1.5, 'kilat': 1.0, 'terjangkau': 1.5, 'worth': 1.5, 
    'simpel': 1.5, 'stylish': 1.5, 'sederhana': 1.5, 'andalan': 1.0, 'gak nyesel': 1.0, 'normal': 1.0, 'kece': 2.0, 
    'istimewa': 2.5, 'sempurna': 2.5, 'prima': 1.0, 'mengagumkan': 1.0, 'indah': 1.0, 'cantik': 1.0, 
    'menakjubkan': 1.0, 'spektakuler': 2.5, 'dahsyat': 1.0, 'fantastis': 2.0, 'luar biasa': 2.5, 'asyik': 1.0, 
    'asik': 1.0, 'juara': 2.0, 'emas': 1.0, 'hebat': 1.0, 'top': 1.0, 'teratas': 1.0, 'unggul': 1.0, 'super': 2.0, 
    'briliant': 1.0, 'handal': 1.0, 'cerdas': 1.0, 'premium': 2.5, 'gemilang': 1.0, 'manis': 1.0, 'kualitas': 1.0, 
    'puass': 1.0, 'ideal': 1.0, 'mulus': 1.0, 'epic': 1.0, 'elegant': 1.0, 'tokcer': 1.0, 'dewa': 1.0, 'ciamik': 1.0, 
    'murmer': 1.0, 'ngga ngecewain': 1.0, 'ga ngecewain': 1.0, 'menarik': 1.5, 'menyenangkan': 1.0, 'menghibur': 1.0, 
    'menyegarkan': 1.0, 'bersih': 1.0, 'praktis': 1.5, 'berguna': 1.5, 'berharga': 1.0, 'tahan lama': 1.5, 
    'awet': 1.0, 'hemat': 1.0, 'ekonomis': 1.0, 'efisien': 1.0, 'jernih': 1.0, 'terampil': 1.0, 'santai': 1.0, 
    'lega': 1.0, 'segar': 1.0, 'sehat': 1.0, 'fit': 1.0, 'bugar': 1.0, 'bermanfaat': 1.0, 'produktif': 1.0, 'kreatif': 1.0, 
    'inovatif': 1.0, 'imajinatif': 1.0, 'bijaksana': 1.0, 'pintar': 1.0, 'cendekia': 1.0, 'ramah': 1.0, 'sopan': 1.0, 
    'memadai': 1.0, 'seimbang': 1.0, 'stabil': 1.0, 'irit': 1.0, 'terpercaya': 1.0, 'jujur': 1.0, 'lengkap': 1.0, 
    'teliti': 1.0, 'beruntung': 1.0, 'pasti': 1.0, 'yakin': 1.0, 'tentu': 1.0, 'berterima kasih': 1.0, 'makasih': 1.0, 
    'thanks': 1.0, 'kenang': 1.0, 'berkesan': 1.0, 'patut': 1.0, 'layak': 1.0, 'mudah': 1.0, 'efektif': 1.0, 'standar': 1.0, 
    'umum': 1.0, 'setuju': 1.0, 'mendukung': 1.0, 'membantu': 1.0, 'sukses': 1.0, 'melindungi': 1.0, 'menjaga': 1.0, 
    'memelihara': 1.0, 'merawat': 1.0, 'menghargai': 1.0, 'mengapresiasi': 1.0, 'menghormati': 1.0, 'percaya': 1.0 
}

# Dictionary kata negatif dengan bobot
neg_words = {
    'jelek': 2.0, 'buruk': 2.0, 'rusak': 2.0, 'kecewa': 2.5, 'tidak': 1.0, 'nggak': 1.0, 'ga': 1.0, 'gak': 1.0, 
    'kurang': 1.5, 'bau': 2.0, 'mahal': 1.5, 'sampah': 2.5, 'cacat': 2.0, 'gembel': 2.0, 'bocor': 2.0, 'busuk': 2.5, 
    'kw': 2.0, 'palsu': 2.0, 'bohong': 2.0, 'robek': 2.0, 'sobek': 2.0, 'titik': 1.0, 'noda': 1.5, 'kotor': 1.5, 
    'lamban': 1.5, 'lambat': 1.5, 'cot': 2.0, 'gembel': 2.0, 'telat': 1.5, 'molor': 1.5, 'belum': 1.0, 'pusing': 1.5, 
    'ribet': 1.5, 'ngga banget': 2.0, 'salah': 1.5, 'gagal': 2.0, 'engga datang': 2.0, 'ditunggu': 1.0, 'abal': 2.0, 
    'dicuci luntur': 2.0, 'ditipu': 2.5, 'keasinan': 1.5, 'kemanisan': 1.5, 'hambar': 1.5, 'keras': 1.5, 'alot': 1.5, 
    'basi': 2.0, 'kemahalan': 1.5, 'bosan': 1.0, 'garing': 1.5, 'monoton': 1.0, 'membosankan': 1.5, 'terbatas': 1.0, 
    'rugi': 2.0, 'penipuan': 2.5, 'sia-sia': 2.0, 'botak': 1.5, 'meleber': 1.5, 'warna luntur': 2.0, 'kekecilan': 1.5, 
    'kebesaran': 1.5, 'sempit': 1.5, 'longgar': 1.0, 'lecet': 1.5, 'penyok': 1.5, 'kempes': 1.5, 'ambigu': 1.0, 
    'tidak jelas': 1.5, 'bingung': 1.0, 'seret': 1.5, 'lambat': 1.5, 'tak layak': 2.0, 'tak pantas': 2.0, 'jamuran': 2.0, 
    'memble': 1.5, 'minder': 1.0, 'pesimis': 1.0, 'malu': 1.0, 'kuno': 1.0, 'ketinggalan zaman': 1.0, 'rumit': 1.5, 
    'tidak berguna': 2.0, 'tidak berharga': 2.0, 'bolos': 1.0, 'mangkir': 1.0, 'teledor': 1.5, 'abai': 1.5, 'ceroboh': 1.5, 
    'sembrono': 1.5, 'ugal-ugalan': 1.5, 'jorok': 2.0, 'kasar': 1.5, 'brutal': 2.0, 'biadab': 2.0, 'kejam': 2.0, 'sengsara': 2.0, 
    'menderita': 2.0, 'susah': 1.5, 'sulit': 1.5, 'tertekan': 1.5, 'sedih': 1.5, 'nestapa': 1.5, 'muram': 1.5, 'putus asa': 2.0, 
    'sepi': 1.0, 'mencurigakan': 1.5, 'ragu': 1.0, 'takut': 1.5, 'cemas': 1.5, 'frustrasi': 2.0, 'marah': 2.0, 'geram': 2.0, 
    'benci': 2.0, 'dendam': 2.0, 'dengki': 2.0, 'iri': 1.5, 'cemburu': 1.5, 'serakah': 1.5, 'tamak': 1.5, 'kikir': 1.5, 
    'pelit': 1.5, 'mengerikan': 2.0, 'menyeramkan': 2.0, 'horor': 2.0, 'mengecewakan': 2.5, 'sial': 2.0, 'celaka': 2.0, 
    'malang': 1.5, 'apes': 1.5, 'naas': 1.5, 'remeh': 1.5, 'pandir': 1.5, 'bebal': 1.5, 'dungu': 1.5, 'bodoh': 1.5, 
    'tolol': 2.0, 'idiot': 2.0, 'dangkal': 1.5, 'konyol': 1.5, 'absurd': 1.5, 'aneh': 1.0, 'ganjil': 1.0, 'merepotkan': 1.5, 
    'mengganggu': 1.5, 'mengesalkan': 1.5, 'menjengkelkan': 1.5, 'menyebalkan': 1.5, 'nakal': 1.0, 'bandel': 1.0, 'degil': 1.0, 
    'keras kepala': 1.0, 'ngotot': 1.0, 'kolot': 1.0, 'primitif': 1.0, 'sakit': 1.5, 'nyeri': 1.5, 'perih': 1.5, 'pedih': 1.5, 
    'pilu': 1.5, 'ngilu': 1.5, 'turun': 1.0, 'jatuh': 1.0, 'runtuh': 1.5, 'hancur': 2.0, 'menurun': 1.0, 'minus': 1.5,
    'tipis': 1.0, 'bolong': 1.5, 'lubang': 1.0, 'beda': 1.0, 'lama': 1.5, 'terlalu': 1.0
}

# Update lexicon VADER dengan kata Bahasa Indonesia
sia.lexicon.update(pos_words)
sia.lexicon.update(neg_words)

def preprocess_text(text):
    """Preprocessing teks: mengubah ke lowercase dan menghapus karakter non-alfanumerik"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Mengubah teks ke lowercase
    text = text.lower()
    
    return text

def process_chunk(chunk):
    """Process a chunk of the DataFrame"""
    results = []
    
    for _, row in chunk.iterrows():
        # Ambil komentar dan lakukan preprocessing
        raw_comment = row['komentar']
        comment = preprocess_text(raw_comment)  # Preprocess teks sebelum analisis
        
        rating = row.get('rating', None)  # Ambil rating jika ada
        
        # Nilai default untuk sentimen
        compound_score = 0
        sentiment_label = 'neutral'
        pos_words_found = []
        neg_words_found = []
        
        # Kasus khusus: Rating 5 tanpa komentar dianggap netral
        if rating == 5 and not comment:
            compound_score = 0  # Netral (0)
            sentiment_label = 'neutral'
        
        # Kasus lainnya: Analisis normal
        else:
            # Analisis teks jika ada komentar
            if comment:
                # Analisis sentimen dengan VADER
                sentiment_scores = sia.polarity_scores(comment)
                text_score = sentiment_scores['compound']
                
                # Ekstrak kata positif dan negatif
                words = comment.split()
                pos_words_found = [word for word in words if word in pos_words]
                neg_words_found = [word for word in words if word in neg_words]
            else:
                text_score = 0  # Netral jika tidak ada komentar
            
            # Gabungkan dengan rating jika tersedia
            if rating is not None and isinstance(rating, (int, float)):
                # Konversi rating 1-5 ke skala -1 hingga 1
                rating_score = (rating - 3) / 2
                
                if not comment:
                    # Jika hanya ada rating tanpa komentar, gunakan rating sebagai penentu utama
                    # TETAPI rating 5 tanpa komentar dianggap netral (dihandle di atas)
                    compound_score = rating_score
                else:
                    # Jika ada komentar & rating, beri bobot lebih pada rating (70% rating, 30% teks)
                    compound_score = (rating_score * 0.7) + (text_score * 0.3)
            else:
                # Jika tidak ada rating, gunakan skor teks saja
                compound_score = text_score
            
            # Tentukan label sentimen
            if compound_score > 0.05:
                sentiment_label = 'positive'
            elif compound_score < -0.05:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
        
        # Konversi skor ke skala 0-10 untuk MongoDB
        normalized_score = (compound_score + 1) * 5  # konversi dari -1 to 1 ke 0 to 10
        
        results.append({
            'score': compound_score,
            'normalized_score': normalized_score, 
            'label': sentiment_label,
            'pos_words': ','.join(pos_words_found),
            'neg_words': ','.join(neg_words_found),
            'preprocessed_comment': comment,
            'has_empty_comment': not bool(comment)  # Flag untuk ulasan tanpa komentar
        })
        
    return results

def analyze_sentiment():
    """Analisis sentimen ulasan"""
    print(f"Membaca file CSV: {input_file}")
    
    # Baca data CSV
    df = pd.read_csv(input_file)
    total_reviews = len(df)
    
    print(f"Menganalisis sentimen untuk {total_reviews} ulasan...")
    
    # Bagi dataframe menjadi beberapa chunk untuk processing paralel
    num_workers = os.cpu_count() or 4
    chunk_size = max(1, total_reviews // num_workers)
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, total_reviews, chunk_size)]
    
    start_time = time.time()
    
    # Proses chunk secara paralel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        chunk_results = list(tqdm(
            executor.map(process_chunk, chunks), 
            total=len(chunks),
            desc="Processing chunks"
        ))
        
        # Flatten results
        for chunk_result in chunk_results:
            results.extend(chunk_result)
    
    # Tambahkan hasil ke DataFrame
    df['sentiment_score'] = [r['score'] for r in results]
    df['skor'] = [r['normalized_score'] for r in results]  # Skor untuk MongoDB (0-10)
    df['sentiment_label'] = [r['label'] for r in results]
    df['label'] = [r['label'] for r in results]  # Duplikat untuk MongoDB
    df['posWords'] = [r['pos_words'] for r in results]
    df['negWords'] = [r['neg_words'] for r in results]
    df['preprocessed_comment'] = [r['preprocessed_comment'] for r in results]
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Analisis selesai dalam {processing_time:.2f} detik")
    print(f"Kecepatan: {total_reviews / processing_time:.2f} ulasan per detik")
    
    # Simpan hasil ke CSV
    print(f"Menyimpan hasil ke {output_file}")
    df.to_csv(output_file, index=False)
    
    # Buat file khusus untuk MongoDB
    create_mongodb_file(df)
    
    return df

def create_mongodb_file(df):
    """Buat file CSV yang diformat khusus untuk MongoDB"""
    # Salin DataFrame
    mongo_df = df.copy()
    
    # Pastikan kolom yang diperlukan ada
    required_fields = {
        'ulasanId': 'Unique identifier',
        'produkId': 'Product ID',
        'komentarUlasan': 'The review text',
        'ratingUlasan': 'Rating score',
        'pengguna': 'User who wrote the review',
        'skor': 'Sentiment score (0-10)',
        'label': 'Sentiment label (positif/netral/negatif)',
        'aspek': 'Aspect analysis',
        'alasan': 'Reason for sentiment score'
    }
    
    # Map ke format MongoDB
    mongo_df['ulasanId'] = mongo_df.apply(lambda row: str(row.name), axis=1)  # Gunakan index sebagai ID
    
    # Jika kolom 'produk_id' ada, gunakan itu
    if 'produk_id' in mongo_df.columns:
        mongo_df['produkId'] = mongo_df['produk_id']
    # Jika tidak ada, coba 'produk'
    elif 'produk' in mongo_df.columns:
        mongo_df['produkId'] = mongo_df['produk']
    # Jika keduanya tidak ada, gunakan placeholder
    else:
        mongo_df['produkId'] = "unknown"
    
    # Salin konten ulasan
    mongo_df['komentarUlasan'] = mongo_df['komentar']
    
    # Salin rating jika ada
    if 'rating' in mongo_df.columns:
        mongo_df['ratingUlasan'] = mongo_df['rating']
    else:
        mongo_df['ratingUlasan'] = None
    
    # Salin pengguna jika ada
    if 'pengguna' in mongo_df.columns:
        mongo_df['pengguna'] = mongo_df['pengguna']
    else:
        mongo_df['pengguna'] = "unknown"
    
    # Buat kolom alasan berdasarkan skor dan kata positif/negatif
    mongo_df['alasan'] = mongo_df.apply(
        lambda row: f"Ulasan mendapat rating {row.get('rating', 'tidak ada')} dan mengandung kata positif: {row['posWords'] or 'tidak ada'}, kata negatif: {row['negWords'] or 'tidak ada'}", 
        axis=1
    )
    
    # Kolom aspek untuk MongoDB (format Map/Dictionary dalam format String JSON)
    def create_aspek_json(row):
        aspek_dict = {
            'harga': {'skor': 5, 'label': 'netral'},
            'kualitas': {'skor': 5, 'label': 'netral'},
            'pengiriman': {'skor': 5, 'label': 'netral'},
            'layanan': {'skor': 5, 'label': 'netral'}
        }
        
        # Sesuaikan aspek berdasarkan kata positif/negatif
        pos_words = row['posWords'].split(',') if row['posWords'] else []
        neg_words = row['negWords'].split(',') if row['negWords'] else []
        
        # Kata-kata yang terkait dengan harga
        if any(word in pos_words for word in ['murah', 'terjangkau', 'worth']):
            aspek_dict['harga'] = {'skor': 8, 'label': 'positif'}
        elif any(word in neg_words for word in ['mahal', 'kemahalan']):
            aspek_dict['harga'] = {'skor': 3, 'label': 'negatif'}
        
        # Kata-kata yang terkait dengan kualitas
        if any(word in pos_words for word in ['bagus', 'berkualitas', 'mantap', 'keren']):
            aspek_dict['kualitas'] = {'skor': 8, 'label': 'positif'}
        elif any(word in neg_words for word in ['jelek', 'buruk', 'rusak']):
            aspek_dict['kualitas'] = {'skor': 3, 'label': 'negatif'}
        
        # Kata-kata yang terkait dengan pengiriman
        if any(word in pos_words for word in ['cepat', 'tepat']):
            aspek_dict['pengiriman'] = {'skor': 8, 'label': 'positif'}
        elif any(word in neg_words for word in ['lambat', 'telat', 'lama']):
            aspek_dict['pengiriman'] = {'skor': 3, 'label': 'negatif'}
        
        # Kata-kata yang terkait dengan layanan
        if any(word in pos_words for word in ['ramah', 'responsif', 'membantu']):
            aspek_dict['layanan'] = {'skor': 8, 'label': 'positif'}
        elif any(word in neg_words for word in ['kasar', 'tidak']):
            aspek_dict['layanan'] = {'skor': 3, 'label': 'negatif'}
        
        return str(aspek_dict)  # Return sebagai string untuk CSV
    
    mongo_df['aspek'] = mongo_df.apply(create_aspek_json, axis=1)
    
    # Pilih kolom yang diperlukan untuk MongoDB
    columns_for_mongodb = ['ulasanId', 'produkId', 'komentarUlasan', 'ratingUlasan', 
                           'pengguna', 'skor', 'label', 'aspek', 'alasan']
    
    # Simpan file untuk MongoDB
    mongo_df[columns_for_mongodb].to_csv(os.path.join(output_dir, mongodb_ready_file), index=False)
    print(f"File untuk import ke MongoDB disimpan di: {os.path.join(output_dir, mongodb_ready_file)}")

def visualize_results(df):
    """Buat visualisasi dari hasil analisis sentimen"""
    # Hitung distribusi label sentimen
    sentiment_counts = df['sentiment_label'].value_counts()
    total_reviews = len(df)
    
    # 1. Pie chart distribusi sentimen
    plt.figure(figsize=(10, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
            colors=['#4CAF50', '#FFC107', '#F44336'])
    plt.title('Distribusi Sentimen Ulasan')
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
    plt.close()
    
    # 2. Analisis kata positif dan negatif yang paling sering muncul
    pos_words_all = ','.join(df['posWords'].dropna()).split(',')
    pos_words_all = [word for word in pos_words_all if word]  # Filter empty strings
    pos_words_count = pd.Series(pos_words_all).value_counts().head(10)

    neg_words_all = ','.join(df['negWords'].dropna()).split(',')
    neg_words_all = [word for word in neg_words_all if word]  # Filter empty strings
    neg_words_count = pd.Series(neg_words_all).value_counts().head(10)
    
    # 3. Visualisasi kata positif teratas
    plt.figure(figsize=(12, 6))
    if not pos_words_count.empty:
        sns.barplot(x=pos_words_count.values, y=pos_words_count.index, palette='Greens_r')
        plt.title('10 Kata Positif Terpopuler')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_positive_words.png'))
    plt.close()
    
    # 4. Visualisasi kata negatif teratas
    plt.figure(figsize=(12, 6))
    if not neg_words_count.empty:
        sns.barplot(x=neg_words_count.values, y=neg_words_count.index, palette='Reds_r')
        plt.title('10 Kata Negatif Terpopuler')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_negative_words.png'))
    plt.close()
    
    # 5. Hubungan antara rating dan sentimen
    if 'rating' in df.columns:
        plt.figure(figsize=(10, 6))
        cross_tab = pd.crosstab(df['rating'], df['sentiment_label'], normalize='index') * 100
        cross_tab.plot(kind='bar', stacked=True, colormap='RdYlGn')
        plt.title('Distribusi Sentimen Berdasarkan Rating')
        plt.xlabel('Rating')
        plt.ylabel('Persentase')
        plt.legend(title='Sentimen')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sentiment_by_rating.png'))
        plt.close()
    
    # 6. Jika ada kolom produk, buat analisis per produk
    if 'produk' in df.columns:
        # Ambil produk dengan jumlah ulasan terbanyak (top 10)
        top_products = df['produk'].value_counts().head(10).index
        
        # Filter ulasan hanya untuk produk teratas
        df_top_products = df[df['produk'].isin(top_products)]
        
        # Hitung sentimen per produk
        sentiment_by_product = pd.crosstab(
            df_top_products['produk'], 
            df_top_products['sentiment_label'],
            normalize='index'
        ) * 100
        
        # Visualisasi sentimen per produk
        plt.figure(figsize=(14, 8))
        sentiment_by_product.plot(kind='barh', stacked=True, colormap='RdYlGn')
        plt.title('Analisis Sentimen per Produk (Top 10)')
        plt.xlabel('Persentase')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sentiment_by_product.png'))
        plt.close()

def main():
    """Fungsi utama program"""
    # Analisis sentimen
    df_results = analyze_sentiment()
    
    # Buat visualisasi
    visualize_results(df_results)
    
    # Tampilkan statistik
    total_reviews = len(df_results)
    positive_reviews = len(df_results[df_results['sentiment_label'] == 'positive'])
    neutral_reviews = len(df_results[df_results['sentiment_label'] == 'neutral'])
    negative_reviews = len(df_results[df_results['sentiment_label'] == 'negative'])
    
    print(f"\nRingkasan Analisis Sentimen:")
    print(f"Total ulasan yang dianalisis: {total_reviews}")
    print(f"Ulasan positif: {positive_reviews} ({positive_reviews/total_reviews*100:.1f}%)")
    print(f"Ulasan netral: {neutral_reviews} ({neutral_reviews/total_reviews*100:.1f}%)")
    print(f"Ulasan negatif: {negative_reviews} ({negative_reviews/total_reviews*100:.1f}%)")
    print(f"\nFile yang dihasilkan:")
    print(f"1. {output_file} - File CSV lengkap")
    print(f"2. {os.path.join(output_dir, mongodb_ready_file)} - File CSV untuk diimpor ke MongoDB")
    print(f"3. {os.path.join(output_dir, 'sentiment_distribution.png')} - Visualisasi distribusi sentimen")
    print(f"4. {os.path.join(output_dir, 'top_positive_words.png')} - Visualisasi kata positif terpopuler")
    print(f"5. {os.path.join(output_dir, 'top_negative_words.png')} - Visualisasi kata negatif terpopuler")

if __name__ == "__main__":
    main()