for f in *.mp3; do
  ffmpeg -i "$f" "${f%.mp3}.wav"
done
