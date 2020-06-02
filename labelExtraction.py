import json
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

import acoustics
import numpy as np
from pydub import AudioSegment
from pydub.generators import WhiteNoise
from pydub.utils import make_chunks

AUDIO_LENGTH = 30
MIN_SEG_LENGTH = 0.25

WN_PCT = [25]

base_audio = 'audio/'
base_labels = 'labels/'
audio_files = os.listdir(base_audio)
label_files_all = os.listdir(base_labels)


def preprocess_file(file):
    print('File {} started'.format(file))
    base_file = base_audio + file

    label_files = [f for f in label_files_all if file in f]
    label_list = []

    label_dict = {}

    for lfile in label_files:
        tree = ET.parse(base_labels + lfile)
        root = tree.getroot()
        for child in root.iter('dialogueact'):
            id = int(child.attrib['{http://nite.sourceforge.net/}id'].split(
                '.')[2].replace('dialogueact', ''))
            starttime = float(child.attrib['starttime'])
            endtime = float(child.attrib['endtime'])
            speaker = child.attrib['participant']
            label_list.append({'id': id, 'start': starttime,
                               'end': endtime, 'speaker': speaker})

    label_list = sorted(label_list, key=lambda label: int(label['id']))
    audio_file = [f for f in os.listdir(base_file) if not 'new_data' in f][0]
    audio_seg = AudioSegment.from_file(base_file + '/' + audio_file)

    extract_audio(label_list, audio_seg, base_file, label_dict, file)


def preprocess_file_cgn(file):
    print('File {} started'.format(file))
    base_file = base_audio + file

    label_files = [f for f in label_files_all if file in f]
    label_list = []

    label_dict = {}

    for lfile in label_files:
        tree = ET.parse(base_labels + lfile)
        root = tree.getroot()
        prev_segment = {}
        for child in root.iter('tau'):
            id = child.attrib['ref'].split('.')[1]
            starttime = child.attrib['tb']
            endtime = child.attrib['te']
            speaker = child.attrib['s']
            segment = {'id': int(id), 'start': float(starttime),
                       'end': float(endtime), 'speaker': speaker}
            if prev_segment == {}:
                prev_segment = segment
            else:
                if prev_segment['end'] == segment['start'] and prev_segment['speaker'] == segment['speaker']:
                    prev_segment['end'] = segment['end']
                else:
                    label_list.append(prev_segment)
                    prev_segment = segment

    label_list = sorted(label_list, key=lambda label: int(label['id']))
    audio_file = [f for f in os.listdir(base_file) if not 'new_data' in f][0]
    audio_seg = AudioSegment.from_file(base_file + '/' + audio_file)

    extract_audio(label_list, audio_seg, base_file, label_dict, file)


def extract_audio(label_list, audio_seg, base_file, label_dict, file):
    unique_speaker = set(map(lambda x: x['speaker'], label_list))

    l = len(audio_seg)

    dir_name = "{}/{}/".format(base_file, 'new_data')

    audio_db = audio_seg.dBFS

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    label_dict['{}.wav'.format(file)] = {
        'labels': label_list, 'no_speakers': len(unique_speaker)}

    with open("{}/{}/{}.json".format(base_file, 'new_data', file), 'w') as outfile:
        json.dump(label_dict, outfile)

    # Export default file

    default_file_name = "{}/{}/{}.wav".format(
        base_file, 'new_data/default', file)

    if not os.path.exists("{}/{}/".format(base_file, 'new_data/default')):
        os.mkdir("{}/{}/".format(base_file, 'new_data/default'))

    audio_seg.export(default_file_name, format='wav')

    # Export white noise file

    for pct in WN_PCT:
        noise_file_name = "{}/{}/{}_noise{}.wav".format(
            base_file, 'new_data/noise' + str(pct), file, pct)

        if not os.path.exists("{}/{}/".format(base_file, 'new_data/noise' + str(pct))):
            os.mkdir("{}/{}/".format(base_file,
                                     'new_data/noise' + str(pct)))

        pct = pct / 100
        wn_db = (1 + (1 - pct)) * audio_db

        noise = WhiteNoise().to_audio_segment(duration=len(audio_seg)).apply_gain(wn_db)
        noise_audio_seg = audio_seg.overlay(noise)
        noise_audio_seg.export(noise_file_name, format='wav')

    # Export random overlay file

    sound_effect_list = os.listdir('sound_fx')

    rnd_fx = sound_effect_list[random.randrange(
        len(sound_effect_list))]

    print(rnd_fx)

    random_effect = AudioSegment.from_file('sound_fx/' + rnd_fx)

    fx_audio_seg = audio_seg.overlay(
        (random_effect).apply_gain(audio_db * 0.4), loop=True)

    fx_file_name = "{}/{}/{}_fx_overlay.wav".format(
        base_file, 'new_data/fx_overlay', file)

    if not os.path.exists("{}/{}/".format(base_file, 'new_data/fx_overlay')):
        os.mkdir("{}/{}/".format(base_file, 'new_data/fx_overlay'))

    fx_audio_seg.export(fx_file_name, format='wav')

    # # Export random overlay + WN file

    # overlay_noise_file_name = "{}/{}/{}_noise_overlay.wav".format(
    #     base_file, 'new_data/noise_overlay', file)

    # if not os.path.exists("{}/{}/".format(base_file, 'new_data/noise_overlay')):
    #     os.mkdir("{}/{}/".format(base_file,
    #                              'new_data/noise_overlay'))

    # wn_db = (1 + (1 - 0.5)) * audio_db

    # noise = WhiteNoise().to_audio_segment(duration=len(audio_seg)).apply_gain(wn_db)
    # fx_audio_seg_noise = fx_audio_seg.overlay(noise)
    # fx_audio_seg_noise.export(overlay_noise_file_name, format='wav')

    # # Well, rip

    # extreme_seg = audio_seg

    # extreme_file_name = "{}/{}/{}_extreme.wav".format(
    #     base_file, 'new_data/extreme', file)

    # if not os.path.exists("{}/{}/".format(base_file, 'new_data/extreme')):
    #     os.mkdir("{}/{}/".format(base_file,
    #                              'new_data/extreme'))

    # for _ in range(3):
    #     rnd_fx = sound_effect_list[random.randrange(
    #         len(sound_effect_list))]

    #     random_effect = AudioSegment.from_file('sound_fx/' + rnd_fx)

    #     extreme_seg = extreme_seg.overlay(
    #         (random_effect * math.ceil(l / len(random_effect))).apply_gain(audio_db * 0.6))

    # wn_db = (1 + (1 - 0.5)) * audio_db
    # noise = WhiteNoise().to_audio_segment(duration=len(audio_seg)).apply_gain(wn_db)
    # extreme_seg = extreme_seg.overlay(noise)
    # extreme_seg.export(extreme_file_name, format='wav')

    print('File {} exported'.format(file))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--cgn':
            base_audio = 'cgn_audio/'
            base_labels = 'cgn_labels/'
            audio_files = os.listdir(base_audio)
            label_files_all = os.listdir(base_labels)
            with ProcessPoolExecutor(max_workers=16) as e:
                e.map(preprocess_file_cgn, audio_files)
    else:
        with ProcessPoolExecutor(max_workers=16) as e:
            e.map(preprocess_file, audio_files)
