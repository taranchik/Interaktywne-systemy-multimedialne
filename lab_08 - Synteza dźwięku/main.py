import json

note_to_halfs = {
    "c": 0,
    "C": 0,
    "cis": 1,
    "Cis": 1,
    "d": 2,
    "D": 2,
    "dis": 3,
    "Dis": 3,
    "e": 4,
    "E": 4,
    "f": 5,
    "F": 5,
    "fis": 6,
    "Fis": 6,
    "g": 7,
    "G": 7,
    "gis": 8,
    "Gis": 8,
    "a": 9,
    "A": 9,
    "ais": 10,
    "Ais": 10,
    "h": 11,
    "H": 11,
}
note_to_halfs["cis"]
note_to_halfs.get("cis")

with open('sound_1.json') as f:
    data = json.load(f)

BaseFreq = data["A4_Freq"]
BPM = data["BPM"]
####
for Line in data["Lines"]:
    ####
    for Note in Line["Notes"]:
        num, den = Note[0].split('/')
        dur = int(num)/int(den)*Tempo
