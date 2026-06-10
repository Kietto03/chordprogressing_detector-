import os
import pandas as pd
import librosa
from yt_dlp import YoutubeDL


def download_audio_from_youtube():
    # 1. Định nghĩa đường dẫn
    raw_dir = "data/raw/McGill-Billboard/annotations/annotations" # Thư mục chứa các folder nhãn 0001, 0002...
    
    if not os.path.exists(raw_dir):
        print(f"⚠️ Không tìm thấy thư mục dữ liệu tại: {raw_dir}. Vui lòng kiểm tra lại cấu trúc folder.")
        return

    # Lấy danh sách tất cả các ID bài hát (tên folder dạng số: 0001, 0002...)
    song_ids = [f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f)) and f.isdigit()]
    song_ids = sorted(song_ids)  # Xóa limit để chạy hết tất cả các bài trong dataset.

    print(f"🎵 Tìm thấy {len(song_ids)} thư mục bài hát cần bổ sung Audio thực tế...")

    # Đọc file index/metadata của McGill Billboard nếu có để tìm tên bài hát chính xác hơn.
    index_path = "data/raw/McGill-Billboard/billboard-2.0-index.csv"
    df_index = None
    if os.path.exists(index_path):
        try:
            df_index = pd.read_csv(index_path)
            print("📊 Đã tải thành công file index/metadata billboard-2.0-index.csv")
        except Exception as e:
            print(f"⚠️ Không thể đọc file index: {e}")
            
    for song_id in song_ids:
        target_folder = os.path.join(raw_dir, song_id)
        
        # Kiểm tra xem folder này đã có file audio (.mp3, .wav, .m4a, .webm) chưa, nếu có rồi thì bỏ qua
        existing_audio = [f for f in os.listdir(target_folder) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.webm'))]
        if existing_audio:
            print(f"⏩ Bài hát {song_id} đã có file audio thực tế. Bỏ qua.")
            continue

        # Cấu hình tìm kiếm và tải xuống của yt-dlp
        # Dùng title và artist từ file index nếu có, nếu không thì dùng tên mặc định
        search_query = f"ytsearch1:McGill Billboard {song_id}"
        if df_index is not None:
            try:
                song_row = df_index[df_index['id'] == int(song_id)]
                if not song_row.empty:
                    title = song_row.iloc[0]['title']
                    artist = song_row.iloc[0]['artist']
                    if pd.notna(title) and pd.notna(artist):
                        search_query = f"ytsearch1:{title} {artist} audio"
                        print(f"🔍 Tra cứu thấy bài {song_id}: '{title}' - {artist}")
            except Exception as e:
                print(f"⚠️ Lỗi khi tra cứu metadata bài {song_id}: {e}")
                
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio',
            'outtmpl': os.path.join(target_folder, 'audio.%(ext)s'), # Ép tên file tải về luôn là 'audio'
            'quiet': True, # Tắt các dòng log thông báo rườm rà của youtube
            'no_warnings': True
        }

        print(f"🚀 Chuyển băng chuyền sang bài {song_id} -> Đang cào nhạc từ YouTube...")
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(search_query, download=True)
                # PR6: Log the actual video ID / title for auditability (helps debug wrong versions)
                if info:
                    video_id = info.get('id', 'unknown')
                    video_title = info.get('title', 'unknown')
                    print(f"   ↳ yt-dlp resolved to: id={video_id} title='{video_title}'")

            # PR6: Post-download validation (duration + basic sanity)
            downloaded = [f for f in os.listdir(target_folder) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.webm'))]
            if downloaded:
                audio_path = os.path.join(target_folder, downloaded[0])
                try:
                    # Quick load to verify it's real audio and has reasonable length
                    y_check, sr_check = librosa.load(audio_path, sr=None, mono=True, duration=10)  # first 10s is enough
                    dur = len(y_check) / sr_check
                    if dur < 5.0:
                        print(f"⚠️  Downloaded audio for {song_id} is very short ({dur:.1f}s). May be a preview or error.")
                    else:
                        print(f"✅ Đã tải thành công audio thật cho bài {song_id} (validated ~{dur:.1f}s)!")
                except Exception as ve:
                    print(f"⚠️  Downloaded file for {song_id} exists but failed librosa validation: {ve}")
            else:
                print(f"❌ Không tìm thấy file audio sau khi tải cho bài {song_id} (Có thể tìm kiếm không thấy kết quả).")
        except Exception as e:
            print(f"❌ Lỗi khi cào nhạc cho bài {song_id}: {e}")

if __name__ == "__main__":
    download_audio_from_youtube()