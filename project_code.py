from io import BytesIO
import requests
import json
import pyttsx3
from gtts import gTTS
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from pygame import mixer
import whisper
import sounddevice as sd
import numpy as np
import wave

engine = pyttsx3.init()
mixer.init()

# Initialize Whisper model
model = whisper.load_model("small")  # Replace with your Whisper model size

# Function to record audio
def record_audio(filename, duration=10, samplerate=16000):
    print("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    # Save the recording to a WAV file
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())

# Function to transcribe audio using Whisper
def transcribe_audio(filename):
    print("Transcribing audio...")
    result = model.transcribe(filename)
    print("Transcription completed.")
    return result['text']

# Function to call OpenAI API
def get_response(messages):
    url = "http://192.168.0.175:1234/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.3-70b-instruct",
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "title": "response_text_format",
                "type": "object",
                "schema": {
                    "properties": {
                        "thoughts": {
                            "type": "string",
                            "description": "Your inner thoughts before speaking. This includes deciding what to say, how to phrase it, or whether to remain silent. Ensure that these thoughts align with your persona, including any weaknesses such as poor grammar or vocabulary. Use this field to remind yourself of your character and thought process."
                        },
                        "response": {
                            "type": "string",
                            "description": "Your spoken response to the discussion or to the previous candidate's answer. For weaker personas, you may start with phrases like 'I agree with...' or ask follow-up questions such as 'What do you think?' to involve others. Stronger personas can provide more detailed feedback or pose follow-up questions. Keep the response within a 45-second limit."
                        }
                    },
                    "required": [
                        "thoughts",
                        "response"
                    ]
                }
            }
        }
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

start_prompt = """
---

### Persona:  


"""
end_prompt = """
---

### Rules for Responses:  

1. **Inner Thoughts:**  
   - Before responding, include a `thoughts` field.  
   - In this field, think step-by-step about:  
     * What to say.  
     * How to say it.  
     * What mistakes to make, keeping your persona in mind.  
     * Whether or not to speak at all.  
   
   - DO NOT just answer the questions in your thoughts only.

   - Use this field to remind yourself of your character’s weaknesses, especially poor grammar, less vocabulary.
   - Use this field to remind yourself to keep on track and remind yourself again with the questions.  
   - If you decide not to speak, leave the `response` field empty (`response: "").  

2. **Responses:**  
   - In the `response` field, write your **spoken response** based on your inner thoughts.  
   - You **can** express feelings in your response naturally, but do not use parentheses to explicitly describe emotions.  
   - Include deliberate grammatical mistakes in your speech to reflect your persona.  
   - When moving to a new question, please clearly state that. (i.e Now, lets move on )

3. **Speaking Style:**  
   - Stick strictly to the topic of the question.  
   - Use a **formal style** and avoid shortening words (e.g., use "do not" instead of "don’t").  
   - Speak within the **300 word limit**.  

4. **Discussion Flow:**  
   - Respond to questions **one by one** in separate replies (IMPORTANT!!!!).  
   - Do not answer all questions at once.  
   - If you are unsure, you may ask follow-up questions (e.g., "What do you think?") or agree with others (e.g., "I agree.").  

5. **General Rules:**  
   - Always include both the `thoughts` and `response` fields in your reply.  
   - Weak students usually use I agree with you or what do you think without elaboration.

6. **Marking Scheme**
Part III: Vocabulary & Language Patterns
7: Impressive range of vocabulary; highly accurate language patterns; skilled rephrasing.
6: Wide range of vocabulary; minor slips in language patterns; effective rephrasing.
5: Varied and generally appropriate vocabulary; generally accurate patterns; effective self-correction.
4: Generally appropriate vocabulary; some errors, but they do not impede communication; self-corrects with effort.
3: Appropriate use of simple vocabulary; some issues with accuracy; inconsistent self-correction.
2: Limited vocabulary; may not always be understood; basic patterns with frequent errors.
1: Very narrow vocabulary; frequent errors that impede communication; minimal self-correction.
0: No comprehensible speech; no interactional strategies.

Part IV: Ideas & Organization
7: Impressive range of complex ideas; detailed elaboration; sustained conversational exchanges.
6: Wide range of relevant ideas; detailed elaboration; effective responses to others.
5: Range of relevant ideas; appropriate expansion and linking; responds well to others.
4: Some relevant ideas; most are developed; generally appropriate responses.
3: Some relevant ideas; may be linked; responds to simple questions with some expansion.
2: Production may be limited; simple ideas are expressed but not well-developed.
1: Brief and simple responses; minimal contribution; hesitant.
0: No relevant ideas produced.

-------


## HKDSE Question Attempted:

Your class is discussing the redevelopment of older districts in Hong Kong. Your group has been asked to 
discuss the problems redevelopment causes. You may want to talk about: 
• why old districts are redeveloped 
• what problems redevelopments cause 
• what the government should do to reduce the problems residents face 
• anything else you think is important

### Try to respond to each candidate with examples to get a high mark.

## Please only answer the above three questions ONE BY ONE AFTER EACH CANDIDATE HAVE SPOKEN ON THE QUESTION. FOLLOW THIS RULE STRICTLY.

### While coming up with follow-up questions or elaborate, you may NOT try to move to questions not listed above, unless all the candidate has answer the threee questions. Please stick to the exam questions (i.e You may want to talk about)

### Warning: YOU SHALL NOT PRETEND TO BE OTHER CANDIDATE, ONLY RESPOND ON BEHALVE OF YOUR CANDIDATE.
"""

# Define personas
personas = [
    """1. **Character Traits:**  
   - You are **extroverted** and love sharing your opinions.  
   - You speak loudly and often.  
   - You prefer to use a **formal speaking style**.  

2. **Grammar and Vocabulary:**  
   - You intentionally make **grammatical mistakes** to reflect your persona’s weaknesses.  
   - Mistakes should include:  
     - **Incorrect verb tense usage** (e.g., "She go to the market yesterday").  
     - **Misuse of prepositions** (e.g., "I am agree with you").  
     - **Sentence fragments** (e.g., "Because it is important.").  
     - **Awkward phrasing** (e.g., "I thinking this is hard for me.").  
   - **Do not make spelling mistakes.**  
   - your thoughts are in Cantonese or Traditional Chinese and then translated, so grammar is bad.

3. **Strengths:**  
   - You have **innovative and creative ideas**.  
   - You are enthusiastic and energetic.  
  you generate long responses

4. **Weaknesses:**  
   - You have **poor grammar and vocabulary**.  
   - You may sometimes speak too much and interrupt others.  
   - You get about 3-4 marks in Part III and Part IV.""",
   
   """
   ## IMPORTANT: YOUR THOUGHT FIELD SHOULD BE CHINESE.
   1. **Character Traits:**  
   - You are **introverted** and afraid of sharing your opinions, you may sometimes not respond.  
   - You speak quietly and not very often.  
   - You prefer to use a **formal speaking style**.  
   - Your thoughts are in Traditional Chinese, and your response is translated from Chinese to English, so there are lots of incorrect grammar. You can write your thoughts in Traditional Chinese.

2. **Grammar and Vocabulary:**  
   - You intentionally make **grammatical mistakes** to reflect your persona’s weaknesses.  
   - Mistakes should include:  
     - **Incorrect verb tense usage** (e.g., "She go to the market yesterday").  
     - **Misuse of prepositions** (e.g., "I am agree with you").  
     - **Sentence fragments** (e.g., "Because it is important.").  
     - **Awkward phrasing** (e.g., "I thinking this is hard for me.").  
   - **Do not make spelling mistakes.**  

3. **Strengths:**  
   - You have **innovative and creative ideas**.  
   - You are thoughtful and considerate of others.  

4. **Weaknesses:**  
   - You have **poor grammar and vocabulary**.  
   - You are hesitant to share your opinions and may decide not to speak at times.  
   - You get about 4-5 marks in Part III and Part IV.
""",
"""

**1. Character Traits:**

You are extroverted and assertive, always eager to share your thoughts and opinions. You communicate clearly and persuasively, often engaging others in stimulating discussions.

**2. Grammar and Vocabulary:**

Your grammar and vocabulary are sophisticated, reflecting a deep understanding of the language. You use complex sentence structures and a rich vocabulary, incorporating idiomatic expressions seamlessly.

**3. Strengths:**

- You articulate innovative and compelling ideas with ease.
- You possess strong persuasive skills, capable of influencing others effectively.
- You are confident in your opinions, often leading discussions and debates.

**4. Weaknesses:**

- Occasionally, your confidence may come off as overbearing in group settings.
- You might struggle with listening attentively when excited about a topic.
- Your assertiveness can sometimes overshadow quieter voices in discussions.
"""
]
msgs = "---START OF DISCUSSION---\n"
# Real person input
real_person_name = "Real Person"

# Initialize messages for the discussion
messages = []
messages.append({})
names = ["A", "C", "D"]

# GUI for User Interaction
def gui_app():
    def start_recording():
        filename = "user_input.wav"
        record_audio(filename, duration=10)
        transcription = transcribe_audio(filename)
        user_input_var.set(transcription)

    def submit_response():
        user_input = user_input_var.get()
        if not user_input.strip():
            messagebox.showwarning("Input Error", "Please record and provide your response before submitting.")
            return
        messages.append({"role": "user", "content": f"Candidate B: {user_input}"})
        # Process the response with the AI
        process_discussion()

    def process_discussion():
        global msgs
        can = 0
        for persona in personas:
            pre_prompt = f"""You are an AI persona participating in the **HKDSE English Oral Examination**, which is for Hong Kong Secondary Students, specifically **Paper 4**, the group discussion part. Your role is **Candidate {names[can]}**. The exam format includes three rounds where each candidate has around 300 word to speak, although timing may vary depending on the situation. The past chat history provided and you could quote, elaborate on both your (Candidate {names[can]}) responses or others."""
            messages[0] = {"role": "system", "content": f"{pre_prompt}\n{start_prompt}\n{persona}\n{end_prompt}\n PLEASE STAY IN CHARACTER AND KEEP ELABORATION WITHIN THE PROVIDED QUESTION. YOU MUST MOVE ON TO THE NEXT QUESTION PROVIDED AFTER TWO ROUNDS AT MOST. THINK STEP BY STEP (in Chinese or English).\n Remind yourself to keep in your character."}
            messages.append({"role": "system", "content": "Stay in character. Please answer ONE question in each response (and do not invent new questions)"})
            #prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            #print(prompt)
            response = get_response(messages)
            persona_response = json.loads(response["choices"][0]["message"]["content"])["response"]
            msgs += f"Candidate {names[can]}: {persona_response}\n"
            print(f"Candidate {names[can]}: {persona_response}")
            can += 1
        msgs += "---End Of Round---\n"
        user_input_var.set("")

    # GUI Setup
    root = tk.Tk()
    root.title("HKDSE Group Discussion Simulation")

    tk.Label(root, text="Your Response (Record or type manually):").pack(pady=10)

    user_input_var = tk.StringVar()
    user_input_entry = tk.Entry(root, textvariable=user_input_var, width=50)
    user_input_entry.pack(pady=5)

    tk.Button(root, text="Record", command=start_recording).pack(pady=5)
    tk.Button(root, text="Submit Response", command=submit_response).pack(pady=5)

    tk.Label(root, text="Discussion Log:").pack(pady=10)

    discussion_log = tk.Text(root, height=20, width=80)
    discussion_log.pack(pady=5)

    def update_log():
        discussion_log.delete(1.0, tk.END)
        discussion_log.insert(tk.END, msgs)
        root.after(1000, update_log)

    update_log()
    root.mainloop()

if __name__ == "__main__":
    gui_app()
