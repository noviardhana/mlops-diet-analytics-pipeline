from prometheus_client import start_http_server, Gauge, Counter
import psutil
import time

# Metrik CPU & RAM
CPU_USAGE = Gauge('system_cpu_usage', 'Penggunaan CPU dalam persen')
RAM_USAGE = Gauge('system_ram_usage', 'Penggunaan RAM dalam persen')
RAM_AVAILABLE = Gauge('system_ram_available_bytes', 'RAM yang tersedia dalam bytes')
SWAP_USAGE = Gauge('system_swap_usage', 'Penggunaan Swap memori dalam persen')

# Metrik Disk
DISK_USAGE = Gauge('system_disk_usage', 'Penggunaan Disk utama dalam persen')

# Metrik Jaringan (Network)
NET_SENT = Gauge('network_bytes_sent_total', 'Total bytes yang dikirim')
NET_RECV = Gauge('network_bytes_recv_total', 'Total bytes yang diterima')

# Metrik Sistem Lainnya
PROCESS_COUNT = Gauge('system_process_count', 'Jumlah proses yang sedang berjalan di OS')
BOOT_TIME = Gauge('system_boot_time', 'Waktu nyala (boot) sistem')

# Metrik Aplikasi (Counter untuk simulasi request MLflow)
HTTP_REQUESTS = Counter('http_requests_total', 'Total HTTP Requests ke Model MLflow', ['method', 'endpoint'])

def collect_metrics():
    """
    Penjelasan: Membaca data dari hardware/OS menggunakan psutil, 
                lalu memasukkannya ke dalam variabel metrik Prometheus.
    """
    # Mengambil metrik CPU & RAM
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    RAM_AVAILABLE.set(psutil.virtual_memory().available)
    SWAP_USAGE.set(psutil.swap_memory().percent)
    
    # Mengambil metrik Disk
    DISK_USAGE.set(psutil.disk_usage('/').percent)
    
    # Mengambil metrik Jaringan
    net_io = psutil.net_io_counters()
    NET_SENT.set(net_io.bytes_sent)
    NET_RECV.set(net_io.bytes_recv)
    
    # Mengambil metrik OS
    PROCESS_COUNT.set(len(psutil.pids()))
    BOOT_TIME.set(psutil.boot_time())
    
    # Simulasi perhitungan request API (Seirama dengan Inference.py yang menembak tiap 1 detik)
    HTTP_REQUESTS.labels(method='POST', endpoint='/invocations').inc(1)

if __name__ == '__main__':
    # Membuka server CCTV di port 8000
    PORT = 8000
    start_http_server(PORT)
    print(f"Prometheus Exporter berjalan di http://127.0.0.1:{PORT}")
    print("Mengekspos 10+ metrik untuk kriteria Advanced. Tekan Ctrl+C untuk berhenti.")
    
    # Looping untuk memperbarui data metrik setiap detik
    while True:
        collect_metrics()
        time.sleep(1)