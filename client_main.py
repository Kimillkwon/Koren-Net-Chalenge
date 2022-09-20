from __future__ import division
import threading as th
import time
import keyboard
from socket import *

keep_going = True
emotion = ""
sttText = ""

def sendEND(end):
    clientSock = socket(AF_INET, SOCK_STREAM)
    clientSock.connect(('10.246.246.65', 13000))

    clientSock.sendall(end.encode('utf-8'))
    time.sleep(1)
    clientSock.close()
    
def recvEmotion():
    global emotion
    serverSock = socket(AF_INET, SOCK_STREAM)
    serverSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    serverSock.bind(('58.238.116.28', 8080))
    serverSock.listen(1)

    connectionSock, addr = serverSock.accept()
    emotionmsg = connectionSock.recv(1024)
    emotion=emotionmsg.decode('utf-8')
    serverSock.close()
    


def sendimg(num):
    import os
    import sys

    clientSock = socket(AF_INET, SOCK_STREAM)
    clientSock.connect(('10.246.246.65', 10000))

    print('연결에 성공했습니다.')
    
    filename = "output"+str(num)+".png"
    print(filename)
    clientSock.sendall(filename.encode('utf-8'))
    time.sleep(1)  #인코딩 하고 디코딩 할수 있도록 시간 벌어주는 용도
    data_transferred = 0


    nowdir = os.getcwd()
    with open(nowdir+"\\"+filename, 'rb') as f: #현재dir에 filename으로 파일을 받는다
        try:
            data = f.read(1024)
            while data: #데이터가 있을 때까지
                data_transferred += clientSock.send(data) #1024바이트 보내고 크기 저장
                data = f.read(1024) #1024바이트 읽음
        except Exception as ex:
            print(ex)
    print("전송완료 %s, 전송량 %d" %(filename, data_transferred))
    clientSock.close()
    

def sendtxt(num):
    global sttText
    text=''
    filename = "output"+str(num)+".txt" 
    ftext = open(filename,'r')
    while True:
        line = ftext.readline()
        if line == '' :
            break
        else:
            text += line.replace("\n",' ')

    ftext.close()
    sttText = text

    clientSock = socket(AF_INET, SOCK_STREAM)
    clientSock.connect(('10.246.246.65', 15000))

    clientSock.sendall(filename.encode('utf-8'))
    time.sleep(1)

    clientSock.send(text.encode('utf-8'))
    clientSock.close()



def key_capture_thread():
    global keep_going
    a = keyboard.read_key()
    if a== "esc":
        #print("esc입력 들어옴")
        keep_going = False
        

def run_quickstart(stt):
    # [START tts_quickstart]
    from google.cloud import texttospeech

    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=stt)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=1.0,
        sample_rate_hertz=16000,
        volume_gain_db=0.0
        )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open("garbage.wav", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        #print('Audio content written to file "output.wav"')
    # [END tts_quickstart]



def make_image(num):
    import glob
    import parselmouth
    import pandas as pd
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    path = os.getcwd()+'/sound/'
    Wavfile_name = os.path.join(
            os.path.dirname(__file__),
            '.',
            "output"+str(num)+".wav")
    snd = parselmouth.Sound(Wavfile_name)
    plt.figure()
    plt.plot(snd.xs(), snd.values.T)
    plt.xlim([snd.xmin, snd.xmax])
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.savefig("output"+str(num)+".png")

def stt(num):
    # [START speech_quickstart]
    import io
    import os
    # The name of the text file to transcribe
    Textfile_name = os.path.join(
        os.path.dirname(__file__),
        '.',
        "output"+str(num)+".txt")
    # text file open('w')
    Textdata=open(Textfile_name,'w')

    # Imports the Google Cloud client library
    # [START migration_import]
    from google.cloud import speech
    # [END migration_import]

    # Instantiates a client
    # [START migration_client]
    client = speech.SpeechClient()
    # [END migration_client]

    # The name of the audio file to transcribe
    file_name = os.path.join(
        os.path.dirname(__file__),
        '.',
        "output"+str(num)+".wav")
    print(file_name)
    data=open(file_name,'rb')


    # Loads the audio into memory
    with io.open(file_name, 'rb') as audio_file:
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        audio_channel_count=1,
        language_code='ko-KR')

    # Detects speech in the audio file
    response = client.recognize(config=config, audio=audio)  # 여기서 api로 넘기는 느낌인데

    for result in response.results:
        text = result.alternatives[0].transcript
        if ('잘가'or '수고해') in text:
            print('Transcript: {}'.format(result.alternatives[0].transcript))
            print("========================")
            print('안녕히 계세요')
            sendEND("true")
            recvEmotion()        
            print("현재 사용자의 감정: "+emotion)
            giveSolution(emotion)
            exit()
        else:
            print('Transcript: {}'.format(result.alternatives[0].transcript))
            print("========================")
            Textdata.write(result.alternatives[0].transcript)
        #run_quickstart(result.alternatives[0].transcript)
    # [END speech_quickstart]
        Textdata.close()

def make_record(num):
    import pyaudio
    import wave
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    WAVE_OUTPUT_FILENAME = "output"+str(num)+".wav"

    p= pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    while keep_going:
        data = stream.read(CHUNK)
        frames.append(data)
        
    #print("esc종료")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


    
def playwav(fileName):
    import pyaudio
    import wave
    fileName = fileName
    chunk = 1024
    path = fileName
    with wave.open(path, 'rb') as f:
        p = pyaudio.PyAudio()
        stream = p.open(format = p.get_format_from_width(f.getsampwidth()), channels = f.getnchannels(), rate = f.getframerate(), output = True)
        data = f.readframes(chunk)
        while data:
            stream.write(data)
            data = f.readframes(chunk)
        stream.stop_stream()
        stream.close()
        p.terminate()


def giveSolution(emotion):
    
    import datetime
    from selenium import webdriver
 
    # Chrome WebDriver를 이용해 Chrome을 실행한다.

    emotion_data = emotion
    
    # 오늘 날짜를 계산한다
    d = str(datetime.datetime.now().day)  
    m = str(datetime.datetime.now().month)
    query = m + '월' + d + '일 멜론'
    query2 = m + '월' + d + '일'
 
    #driver.get("https://www.youtube.com/results?search_query=" + query)
    #driver.get("https://www.youtube.com/results?search_query=" + emotion +"할때 도움되는" + thing)

    if emotion_data == '행복':
        playwav('happySolution.wav')
    elif emotion_data == '우울':

        run_quickstart("당신의 감정은 " + emotion_data + ' 입니다 어떻게 도와 드릴까요?')
        playwav('garbage.wav')

        gsp = Gspeech()
        stt = gsp.getText()
        print(stt)
        if ('음악') in stt:
            thing = '음악'
        elif ('노래') in stt:
            thing = '음악'
        elif ('글귀') in stt:
            thing = '조언'
        elif ('명언') in stt:
            thing = '조언'
        elif ('조언') in stt:
            thing = '조언'
        
        
        run_quickstart("요청하신 "+ emotion_data + "할때 도움되는" + thing + "들려드리도록 하겠습니다")
        playwav('garbage.wav')
        driver = webdriver.Chrome("chromedriver")
        
        if thing == "조언":
            driver.get("https://www.youtube.com/channel/UC0Ra5o9VIVNBpV7vK5G1tIw/search?query=" + emotion_data)
            time.sleep(1)
            continue_link = driver.find_element_by_partial_link_text(emotion_data)
            continue_link.click()
            time.sleep(60)
        elif thing == "음악":
            driver.get("https://www.youtube.com/results?search_query=" + emotion_data +"할때 도움되는" + thing)
            time.sleep(1)
            continue_link = driver.find_element_by_partial_link_text(thing)
            continue_link.click()
            time.sleep(60)
        else:
            print("==========================================")
        
        driver.quit()
    # 감정이 중립일 때
    else:
        playwav('nuetralSolution.wav')


#streamimgSTT
import re
import sys

from google.cloud import speech
import pyaudio
from six.moves import queue
from threading import Thread
import wave

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True
        self.isPause = False

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()


        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def pause(self):
        if self.isPause == False:
            self.isPause = True


    def resume(self):
        if self.isPause == True:
            self.isPause = False


    def status(self):
        return self.isPause

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        if self.isPause == False:
            self._buff.put(in_data)
        #else
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return

            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)


# [END audio_stream]



class Gspeech(Thread):
    def __init__(self):
        Thread.__init__(self)

        self.language_code = 'ko-KR'  # a BCP-47 language tag

        self._buff = queue.Queue()

        self.client = speech.SpeechClient()
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=self.language_code)
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True)

        self.mic = None
        self.status = True

        self.daemon = True
        self.start()

    def __eixt__(self):
        self._buff.put(None)

    def run(self):
        print("run mic")
        with MicrophoneStream(RATE, CHUNK) as stream:
            self.mic = stream
            audio_generator = stream.generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator)

            responses = self.client.streaming_recognize(self.streaming_config, requests)

            # Now, put the transcription responses to use.
            self.listen_print_loop(responses, stream)
        self._buff.put(None)
        self.status = False

    def pauseMic(self):
        if self.mic is not None:
            self.mic.pause()

    def resumeMic(self):
        if self.mic is not None:
            self.mic.resume()

    # 인식된 Text 가져가기
    def getText(self, block = True):
        return self._buff.get(block=block)

    # 음성인식 처리 루틴
    def listen_print_loop(self, responses, mic):
        num_chars_printed = 0
        try:
            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript
                overwrite_chars = ' ' * (num_chars_printed - len(transcript))
                if not result.is_final:
                    sys.stdout.write(transcript + overwrite_chars + '\r')
                    sys.stdout.flush()
                    #### 추가 ### 화면에 인식 되는 동안 표시되는 부분.
                    num_chars_printed = len(transcript)
                else:
                    # 큐에 넣는다.
                    self._buff.put(transcript+overwrite_chars)
                    num_chars_printed = 0
        except:
            return

#chatbot
import os
import sys
sys.path.append('/Users/user/speech/dialogLM')
import numpy as np
import torch
from model.kogpt2 import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer

def chatbot(data):
   
    #sent = input('Question: ')  # '요즘 기분이 우울한 느낌이에요'
    sent = data
    tokenized_indexs = tokenizer.encode(sent)

    input_ids = torch.tensor([tokenizer.bos_token_id,]  + tokenized_indexs +[tokenizer.eos_token_id]).unsqueeze(0)

    # set top_k to 50
    sample_output = model.generate(input_ids=input_ids)
    answer = tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs)+1:], skip_special_tokens = True).split('.')
    '''
    real_answer= []
    for a in answer:
        a = a.split("?")
        real_answer.append(a)
    
    return real_answer[0]
    '''
    return answer[0]

    
def main():

    global keep_going
    global emotion
    global sttText
    num=1

    #이거를 챗봇에서 솔루션을 제공하는 시점에 종료로 수정 필요

    try:
        while True:
            th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
            if num == 1:
                playwav('output.wav') #처음엔 "오늘 하루 어떠셨나요?"  다음부터는 챗봇에서 받아온 텍스트 출력
            #th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
            #playwav('output.wav') #처음엔 "오늘 하루 어떠셨나요?"  다음부터는 챗봇에서 받아온 텍스트 출력
            make_record(num)
            make_image(num)
            stt(num)
                     
            sendEND("false")
            sendimg(num)
            time.sleep(1)
            sendtxt(num)
            #time.sleep(1)
            
            num = num + 1
            keep_going = True

            #여기서 챗봇에서 만들어준 텍스트 tts를 통해서 음성파일 출력  미구현
            run_quickstart(chatbot(sttText))
            playwav('garbage.wav')
            
    except KeyboardInterrupt:
        sendEND("true")
        recvEmotion()        
        print("현재 사용자의 감정: "+emotion)
        giveSolution(emotion)
        print ("c종료")


if __name__ == '__main__':
    #chatbot로드
    root_path = 'C:/Windows/System32/Speech/dialogLM'
    data_path = f"{root_path}/data/wellness_dialog_for_autoregressive_train.txt"
    checkpoint_path =f"{root_path}/checkpoint"
    save_ckpt_path = f"{checkpoint_path}/kogpt2-wellnesee-auto-regressive.pth"

    ctx = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(ctx)

    # 저장한 Checkpoint 불러오기
    checkpoint = torch.load(save_ckpt_path, map_location=device)

    model = DialogKoGPT2()
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    tokenizer = get_kogpt2_tokenizer()
    
    main()


