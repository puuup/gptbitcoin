from youtube_transcript_api import YouTubeTranscriptApi

video_id = "TWINrTppUl4"

# 자막 가져오기
transcript = YouTubeTranscriptApi().fetch(video_id)

# text만 추출해서 출력
for snippet in transcript.snippets:
    print(snippet.text)

# 만약 한 줄로 합치고 싶으면:
all_text = " ".join(snippet.text for snippet in transcript.snippets)
print("\n=== 전체 자막 모음 ===\n")
print(all_text)
