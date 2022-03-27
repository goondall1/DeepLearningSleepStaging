from numpy.core._multiarray_umath import ndarray
import xml.etree.ElementTree as ET
import numpy as np

def sleep_extract_30s_epochs(path):
    try:
        root = ET.parse(path).getroot()
        events = [x for x in list(root.iter()) if x.tag == "ScoredEvent"]
        events_decomposed = list([list(event.iter()) for event in events])
        stage_events = [x for x in events_decomposed if x[1].text == "Stages|Stages"]
        starts = np.array([float(x[3].text) for x in stage_events])/30
        durations = np.array([float(x[4].text) for x in stage_events])/30
        stages = np.array([int(x[2].text[-1]) for x in stage_events])
        stages[stages == 5] = 4
        sleep_timeline: ndarray = np.zeros(int(starts[-1] + durations[-1]))
        for i in range(len(stages)):
            sleep_timeline[int(starts[i]): int(starts[i] + durations[i])] = stages[i]
        return sleep_timeline
    except:
        print(f"Could not extract sleep from: {path}")
        return 0

def sleep_extract(path):
    try:
        root = ET.parse(path).getroot()
        events = [x for x in list(root.iter()) if x.tag == "ScoredEvent"]
        events_decomposed = list([list(event.iter()) for event in events])
        stage_events = [x for x in events_decomposed if x[1].text == "Stages|Stages"]
        starts = np.array([float(x[3].text) for x in stage_events])
        durations = np.array([float(x[4].text) for x in stage_events])
        stages = np.array([int(x[2].text[-1]) for x in stage_events])
        stages[stages == 5] = 4
        sleep_timeline: ndarray = np.zeros(int(starts[-1] + durations[-1]))

        for i in range(len(stages)):
            sleep_timeline[int(starts[i]): int(starts[i] + durations[i])] = stages[i]
        return sleep_timeline
    except:
        print(f"Could not extract sleep from: {path}")
        return 0

def sleep_extract_cleaned(path):
    try:
        root = ET.parse(path).getroot()
        events = [x for x in list(root.iter()) if x.tag == "ScoredEvent"]
        events_decomposed = list([list(event.iter()) for event in events])
        stage_events = [x for x in events_decomposed if x[1].text == "Stages|Stages"]
        starts = np.array([float(x[3].text) for x in stage_events])
        durations = np.array([float(x[4].text) for x in stage_events])
        stages = np.array([int(x[2].text[-1]) for x in stage_events])
        stages[stages == 5] = 4
        for i in np.arange(stages.shape[0]) :
            if durations[i] < 60: #stages[i] != stages[i+1] and stages[i] == stages[i+2] and durations[i] < 90:
                stages[i] = stages[i-1]
        sleep_timeline: ndarray = np.zeros(int(starts[-1] + durations[-1]))

        for i in range(len(stages)):
            sleep_timeline[int(starts[i]): int(starts[i] + durations[i])] = stages[i]
        return sleep_timeline
    except:
        print(f"Could not extract sleep from: {path}")
        return 0

