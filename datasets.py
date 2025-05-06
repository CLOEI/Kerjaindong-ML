from dotenv import load_dotenv
import anthropic
import csv
import os
import time

load_dotenv()

CLAUDE_KEY = os.getenv("CLAUDE_KEY")

client = anthropic.Anthropic(api_key=CLAUDE_KEY)

def generate_data(id_start, request_count=100):
    all_data = []
    
    for i in range(request_count):
        print(f"Request ke-{i + 1} dari {request_count}...")
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=20000,
            temperature=1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Tolong buatkan 20 baris data proyek dalam bentuk tabel dengan format seperti ini, tetapi tanpa header:\n\nID\tDeskripsi Klien\tPlatform\tTujuan Proyek\tFitur Utama\tTarget User\tEstimasi Deadline\tBudget\tIndustri\tButuh UI/UX\tButuh Hosting\n\nBerikut ketentuan tambahan:\n- Kolom *Tujuan Proyek* harus salah satu dari daftar ini: Website, Company Profile, E-Commerce, Landing Page, Sistem Backend, Integrasi Payment, API, Aplikasi Mobile, Aplikasi Desktop, Automasi, Tool Berbasis Web/Software, Design UI/UX.\n- Budget harus masuk akal (misalnya 500000 – 30000000 tergantung kompleksitas proyek) tanpa simbol Rp, langsung angka.\n- Format keluaran langsung dalam bentuk tabel TSV (tab-separated values) tanpa penjelasan tambahan dan tanpa header.\n- Deskripsi Klien harus terdengar alami seperti permintaan nyata dari klien.\n- Estimasi Deadline dalam format “x hari”.\n- Butuh UI/UX dan Butuh Hosting hanya isi \"Ya\" atau \"Tidak\".\n\nMulai dari ID {{id}} dan lanjutkan ke bawah.\n"
                        }
                    ]
                }
            ]
        )

        data = message.content[0].text.strip().split('\n')

        for row in data:
            all_data.append(row.split('\t'))

        id_start += 10

        time.sleep(2)

    return all_data


def save_to_csv(data, filename="dataset.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        writer.writerows(data)

    print(f"Data telah disimpan di {filename}")


generated_data = generate_data(2001, 50)
save_to_csv(generated_data)
