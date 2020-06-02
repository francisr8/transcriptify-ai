import json
import math
import os

import numpy as np
import sklearn.cluster as cluster
import torch
from orderedset import OrderedSet
from pyannote.audio.utils.signal import Binarize, Peak
from pyannote.core import Annotation, Segment, SlidingWindowFeature, Timeline
from pyannote.metrics.detection import DetectionPrecision, DetectionRecall
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.segmentation import (SegmentationCoverage,
                                           SegmentationPurity)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score

pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')

# speech activity detection model trained on AMI training set
sad = torch.hub.load('pyannote/pyannote-audio', 'sad_ami')
# speaker change detection model trained on AMI training set
scd = torch.hub.load('pyannote/pyannote-audio', 'scd_ami')
# overlapped speech detection model trained on AMI training set
# ovl = torch.hub.load('pyannote/pyannote-audio', 'ovl_ami')
# speaker embedding model trained on AMI training set
emb = torch.hub.load('pyannote/pyannote-audio', 'emb_ami')

clusterings = ['MiniBatchKMeans',
               'SpectralClustering', 'AgglomerativeClustering']


def plot_ready(scores): return SlidingWindowFeature(
    np.exp(scores.data[:, 1:]), scores.sliding_window)


def auto_diarization(path):
    diarization = pipeline({'audio': path})
    data = convert_diarization(diarization)
    return data


def select_cluster_algorithm(algorithm, no_clusters):
    if algorithm == 'SpectralClustering':
        return cluster.SpectralClustering(n_clusters=no_clusters)
    elif algorithm == 'MiniBatchKMeans':
        return cluster.MiniBatchKMeans(n_clusters=no_clusters)
    elif algorithm == 'AgglomerativeClustering':
        return cluster.AgglomerativeClustering(n_clusters=no_clusters)


def predict(audio, algorithm='SpectralClustering'):
    # Speech Activation Detection

    sad_scores = sad(audio)
    binarize_sad = Binarize(offset=0.52, onset=0.52, log_scale=True,
                            min_duration_off=0.1, min_duration_on=0.1)
    speech = binarize_sad.apply(sad_scores, dimension=1)

    # Speaker Change Detection

    scd_scores = scd(audio)
    peak = Peak(alpha=0.10, min_duration=0.10, log_scale=True)
    partition = peak.apply(scd_scores, dimension=1)

    # Overlapped Speech Detection

    # ovl_scores = ovl(audio)
    # binarize_ovl = Binarize(offset=0.55, onset=0.55, log_scale=True,
    #                         min_duration_off=0.1, min_duration_on=0.1)
    # overlap = binarize_ovl.apply(ovl_scores, dimension=1)

    # Speaker Embedding

    speech_turns = partition.crop(speech)
    embeddings = emb(audio)

    long_turns = Timeline(
        segments=[s for s in speech_turns if s.duration > .5])

    return long_turns, sad_scores, scd_scores, embeddings


def cluster_annotation(long_turns, embeddings, speakers, algorithm='SpectralClustering'):
    X = []
    for segment in long_turns:
        # "strict" only keeps embedding strictly included in segment
        x = embeddings.crop(segment, mode='strict')
        # average speech turn embedding
        X.append(np.mean(x, axis=0))

    X = np.vstack(X)

    # apply PCA on embeddings
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp.fit_transform(X)

    if (X.shape[1] == 0):
        return Annotation(), [], []

    no_clusters = int(speakers)

    if no_clusters == 0:
        range_n_clusters = list(range(2, 10))

        silhouette_dict = {}

        for n_clusters in range_n_clusters:
            clusterer = cluster.SpectralClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)
            silhouette_dict[n_clusters] = silhouette_avg

        if (all(value == 0 for value in silhouette_dict.values())):
            no_clusters = 2
        else:
            max_val = 0
            max_index = 0
            for clusters in silhouette_dict:
                if (silhouette_dict[clusters] > max_val):
                    max_val = silhouette_dict[clusters]
                    max_index = clusters

            no_clusters = max_index

    c = select_cluster_algorithm(algorithm, no_clusters)
    labels = c.fit_predict(X)

    labeled_data = []
    for i, turn in enumerate(long_turns):
        labeled_data.append([labels[i], turn])

    annotation = Annotation()
    for i in labeled_data:
        label = int(i[0])
        segment = i[1]
        annotation[segment] = label

    return annotation


def custom_diarization(path, speakers):
    audio = {'uri': 'temp.wav', 'audio': path}

    annotation, sad_scores, scd_scores, embeddings = predict(audio)
    annotation = cluster_annotation(annotation, embeddings, speakers)

    return {'annotation': convert_annotation(annotation), 'sad': plot_ready(sad_scores).data.tolist(), 'scd': plot_ready(scd_scores).data.tolist()}


def convert_diarization(diarization):
    diarization = diarization.rename_labels(generator='int')
    return convert_annotation(diarization)


def convert_annotation(diarization):
    data = []
    diarization = diarization.support(0.5)
    for segment, _, label in diarization.itertracks(yield_label=True):
        startNanos = math.floor(
            float(str(segment.start-int(segment.start))[1:]) * 1000000000)
        endNanos = math.floor(
            float(str(segment.end-int(segment.end))[1:]) * 1000000000)
        curr_seg = {
            'startTime': {'seconds': math.floor(segment.start),
                          'nanos': startNanos},
            'endTime': {'seconds': math.floor(segment.end),
                        'nanos': endNanos},
            'word': '',
            'confidence': 0,
            'speakerTag': label + 1
        }
        data.append(curr_seg)
    return data


def test(custom=True, prefix=''):
    der = DiarizationErrorRate(collar=0.5)
    prec = DetectionPrecision(collar=0.5)
    recall = DetectionRecall(collar=0.5)
    coverage = SegmentationCoverage()
    purity = SegmentationPurity()

    result = {}
    if os.path.exists('results.json'):
        with open('results.json') as json_file:
            result = json.load(json_file)
    base_test = prefix + 'audio/'
    test_files = os.listdir(base_test)

    test_path = base_test + test_files[0] + '/new_data/'
    test_types = [name for name in os.listdir(
        test_path) if os.path.isdir(os.path.join(test_path, name))]

    result_data = []
    if (custom):
        for _ in clusterings:
            result_data.append([])

    for test in test_types:
        avg_der = 0
        avg_prec = 0
        avg_rec = 0
        avg_cov = 0
        avg_pur = 0
        counter = 0

        speaker_results = {}
        cluster_results = []
        speaker_results_cluster = []

        for _ in clusterings:
            cluster_results.append({
                'der': 0,
                'prec': 0,
                'rec': 0,
                'cov': 0,
                'pur': 0
            })

            speaker_results_cluster.append({})

        for f in test_files:
            test_file = base_test + f
            data_file = test_file + '/new_data/' + f + '.json'
            with open(data_file) as f:
                data = json.load(f)
                for sub_f in data:
                    counter += 1
                    sub_f_data = data[sub_f]
                    true_labels = sub_f_data['labels']
                    true_speakers = sub_f_data['no_speakers']
                    speakers_int = OrderedSet(
                        map(lambda x: x['speaker'], true_labels))

                    for i, s in enumerate(speakers_int):
                        for datadict in true_labels:
                            if datadict['speaker'] == s:
                                datadict['speaker'] = i + 1

                    true_annotation = convert_to_annotation(true_labels)

                    pred_path = test_file + '/new_data/' + test + '/'
                    pred_file = sub_f if test == 'default' else sub_f.split('.')[
                        0] + '_' + test + '.wav'
                    audio = {'uri': pred_file, 'audio': pred_path + pred_file}

                    if (custom):
                        long_turns, _, _, embeddings = predict(audio)
                        index = 0
                        for algorithm in clusterings:
                            if (custom):
                                pred_annotation = cluster_annotation(
                                    long_turns, embeddings, true_speakers, algorithm)
                            if (type(pred_annotation) is tuple or pred_annotation == Annotation()):
                                continue
                            pred_annotation = pred_annotation.rename_labels(
                                generator='int')

                            der_res = der(true_annotation, pred_annotation)
                            prec_res = prec(true_annotation, pred_annotation)
                            rec_res = recall(true_annotation, pred_annotation)
                            cov_res = coverage(
                                true_annotation, pred_annotation)
                            pur_res = purity(true_annotation, pred_annotation)

                            cluster_results[index]['der'] += der_res
                            cluster_results[index]['prec'] += prec_res
                            cluster_results[index]['rec'] += rec_res
                            cluster_results[index]['cov'] += cov_res
                            cluster_results[index]['pur'] += pur_res

                            if not true_speakers in speaker_results_cluster[index]:
                                speaker_results_cluster[index][true_speakers] = {
                                    'der': 0,
                                    'prec': 0,
                                    'rec': 0,
                                    'cov': 0,
                                    'pur': 0,
                                    'counter': 0
                                }

                            speaker_results_cluster[index][true_speakers]['der'] += der_res
                            speaker_results_cluster[index][true_speakers]['prec'] += prec_res
                            speaker_results_cluster[index][true_speakers]['rec'] += rec_res
                            speaker_results_cluster[index][true_speakers]['cov'] += cov_res
                            speaker_results_cluster[index][true_speakers]['pur'] += pur_res
                            speaker_results_cluster[index][true_speakers]['counter'] += 1

                            index += 1
                    else:
                        pred_annotation = pipeline(
                            {'audio': pred_path + pred_file})

                        der_res = der(true_annotation, pred_annotation)
                        prec_res = prec(true_annotation, pred_annotation)
                        rec_res = recall(true_annotation, pred_annotation)
                        cov_res = coverage(true_annotation, pred_annotation)
                        pur_res = purity(true_annotation, pred_annotation)

                        avg_der += der_res
                        avg_prec += prec_res
                        avg_rec += rec_res
                        avg_cov += cov_res
                        avg_pur += pur_res

                        if not true_speakers in speaker_results:
                            speaker_results[true_speakers] = {
                                'der': 0,
                                'prec': 0,
                                'rec': 0,
                                'cov': 0,
                                'pur': 0,
                                'counter': 0
                            }

                        speaker_results[true_speakers]['der'] += der_res
                        speaker_results[true_speakers]['prec'] += prec_res
                        speaker_results[true_speakers]['rec'] += rec_res
                        speaker_results[true_speakers]['cov'] += cov_res
                        speaker_results[true_speakers]['pur'] += pur_res
                        speaker_results[true_speakers]['counter'] += 1
        if custom:
            index = 0
            for algorithm in clusterings:
                cluster_data = cluster_results[index]
                sub_data = {'type': test}
                sub_data['DER'] = cluster_data['der'] / counter
                sub_data['Precision'] = cluster_data['prec'] / counter
                sub_data['Recall'] = cluster_data['rec'] / counter
                sub_data['Coverage'] = cluster_data['cov'] / counter
                sub_data['Purity'] = cluster_data['pur'] / counter

                for s in speaker_results_cluster[index]:
                    speaker_results_cluster[index][s]['der'] = speaker_results_cluster[index][s]['der'] / \
                        speaker_results_cluster[index][s]['counter']
                    speaker_results_cluster[index][s]['prec'] = speaker_results_cluster[index][s]['prec'] / \
                        speaker_results_cluster[index][s]['counter']
                    speaker_results_cluster[index][s]['rec'] = speaker_results_cluster[index][s]['rec'] / \
                        speaker_results_cluster[index][s]['counter']
                    speaker_results_cluster[index][s]['cov'] = speaker_results_cluster[index][s]['cov'] / \
                        speaker_results_cluster[index][s]['counter']
                    speaker_results_cluster[index][s]['pur'] = speaker_results_cluster[index][s]['pur'] / \
                        speaker_results_cluster[index][s]['counter']

                sub_data['Speaker_data'] = speaker_results_cluster[index]

                result_data[index].append(sub_data)
                result[prefix + 'custom' + algorithm] = result_data[index]
                index += 1
        else:
            sub_data = {'type': test}
            sub_data['DER'] = avg_der / counter
            sub_data['Precision'] = avg_prec / counter
            sub_data['Recall'] = avg_rec / counter
            sub_data['Coverage'] = avg_cov / counter
            sub_data['Purity'] = avg_pur / counter

            for s in speaker_results:
                speaker_results[s]['der'] = speaker_results[s]['der'] / \
                    speaker_results[s]['counter']
                speaker_results[s]['prec'] = speaker_results[s]['prec'] / \
                    speaker_results[s]['counter']
                speaker_results[s]['rec'] = speaker_results[s]['rec'] / \
                    speaker_results[s]['counter']
                speaker_results[s]['cov'] = speaker_results[s]['cov'] / \
                    speaker_results[s]['counter']
                speaker_results[s]['pur'] = speaker_results[s]['pur'] / \
                    speaker_results[s]['counter']

            sub_data['Speaker_data'] = speaker_results

            result_data.append(sub_data)
            result[prefix + 'auto'] = result_data

        save_file = 'results.json'

        with open(save_file, 'w') as outfile:
            json.dump(result, outfile)
    return result_data


def convert_to_annotation(labels):
    hypothesis = Annotation()
    for l in labels:
        hypothesis[Segment(l['start'], l['end'])] = l['speaker']
    return hypothesis
