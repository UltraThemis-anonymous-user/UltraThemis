import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

class soundBase:
    def __init__(self, path):
        self.path = path

    def audioread(self, formater='sample'):
        fs, data = wavfile.read(self.path)
        bits = 1
        if type(data[0]) is np.int16:
            bits = 16
        if formater == 'sample':
            data = data / (2 ** (bits - 1))
        return data, fs

    def SPL(self, data, fs, frameLen=100, isplot=True):

        def spl_cal(s, fs, frameLen):

            l = len(s)
            M = frameLen * fs / 1000
            if not l == M:
                exit('the length of frame and signal is not equal!')
            # 计算有效声压
            pp = 0
            for i in range(int(M)):
                pp += (s[i] * s[i])
            pa = np.sqrt(pp / M)
            p0 = 2e-5
            spl = 20 * np.log10(pa / p0)
            return spl

        length = len(data)
        M = fs * frameLen // 1000
        m = length % M
        if not m < M // 2:
            data = np.hstack((data, np.zeros(M - m)))
        else:
            data = data[:M * (length // M)]
        spls = np.zeros(len(data) // M)
        for i in range(length // M - 1):
            s = data[i * M:(i + 1) * M]
            spls[i] = spl_cal(s, fs, frameLen)

        if isplot:
            plt.subplot(211)
            plt.plot(data)
            plt.subplot(212)
            plt.step([i for i in range(len(spls))], spls)
            plt.show()
        return spls
def FrameTimeC(frameNum, frameLen, inc, fs):
    ll = np.array([i for i in range(frameNum)])
    return ((ll - 1) * inc + frameLen / 2) / fs
def enframe(x, win, inc=None):
    nx = len(x)
    if isinstance(win, list):
        nwin = len(win)
        nlen = nwin  
    elif isinstance(win, np.ndarray):
        nwin = len(win)
        nlen = nwin 
    elif isinstance(win, int):
        nwin = 1
        nlen = win  
    if inc is None:
        inc = nlen
    nf = (nx - nlen + inc) // inc
    frameout = np.zeros((nf, nlen))
    indf = np.multiply(inc, np.array([i for i in range(nf)]))
    for i in range(nf):
        frameout[i, :] = x[indf[i]:indf[i] + nlen]
    if isinstance(win, list):
        frameout = np.multiply(frameout, np.array(win))
    return frameout
def vad_revr(dst1, T1, T2):
    """
    :param dst1:
    :param T1:
    :param T2:
    :return:
    """
    fn = len(dst1)
    maxsilence = 8
    minlen = 5
    status = 0
    count = np.zeros(fn)
    silence = np.zeros(fn)
    xn = 0
    x1 = np.zeros(fn)
    x2 = np.zeros(fn)
    for n in range(1, fn):
        if status == 0 or status == 1:
            if dst1[n] < T2:
                x1[xn] = max(1, n - count[xn] - 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif dst1[n] < T1:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        if status == 2:
            if dst1[n] < T1:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < maxsilence:
                    count[xn] += 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        if status == 3:
            status = 0
            xn += 1
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    el = len(x1[:xn])
    if x1[el - 1] == 0:
        el -= 1
    if x2[el - 1] == 0:
        print('Error: Not find endding point!\n')
        x2[el] = fn
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1
        NF[int(x1[i]):int(x2[i])] = 0
    voiceseg = findSegment(np.where(SF == 1)[0])
    vsl = len(voiceseg.keys())
    return voiceseg, vsl, SF, NF


def findSegment(express):
    """
    :param express:
    :return:
    """
    if express[0] == 0:
        voiceIndex = np.where(express)
    else:
        voiceIndex = express
    d_voice = np.where(np.diff(voiceIndex) > 1)[0]
    voiceseg = {}
    if len(d_voice) > 0:
        for i in range(len(d_voice) + 1):
            seg = {}
            if i == 0:
                st = voiceIndex[0]
                en = voiceIndex[d_voice[i]]
            elif i == len(d_voice):
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[-1]
            else:
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[d_voice[i]]
            seg['start'] = st
            seg['end'] = en
            seg['duration'] = en - st + 1
            voiceseg[i] = seg
    return voiceseg


def vad_specEN(data, wnd, inc, NIS, thr1, thr2, fs):
    import matplotlib.pyplot as plt
    from scipy.signal import medfilt
    x = enframe(data, wnd, inc)
    X = np.abs(np.fft.fft(x, axis=1))
    if len(wnd) == 1:
        wlen = wnd
    else:
        wlen = len(wnd)
    df = fs / wlen
    fx1 = int(250 // df + 1)  
    fx2 = int(3500 // df + 1)  
    km = wlen // 8
    K = 0.5
    E = np.zeros((X.shape[0], wlen // 2))
    E[:, fx1 + 1:fx2 - 1] = X[:, fx1 + 1:fx2 - 1]
    E = np.multiply(E, E)
    Esum = np.sum(E, axis=1, keepdims=True)
    P1 = np.divide(E, Esum)
    E = np.where(P1 >= 0.9, 0, E)
    Eb0 = E[:, 0::4]
    Eb1 = E[:, 1::4]
    Eb2 = E[:, 2::4]
    Eb3 = E[:, 3::4]
    Eb = Eb0 + Eb1 + Eb2 + Eb3
    prob = np.divide(Eb + K, np.sum(Eb + K, axis=1, keepdims=True))
    Hb = -np.sum(np.multiply(prob, np.log10(prob + 1e-10)), axis=1)
    for i in range(10):
        Hb = medfilt(Hb, 5)
    Me = np.mean(Hb)
    eth = np.mean(Hb[:NIS])
    Det = eth - Me
    T1 = thr1 * Det + Me
    T2 = thr2 * Det + Me
    voiceseg, vsl, SF, NF = vad_revr(Hb, T1, T2)
    return voiceseg, vsl, SF, NF, Hb

def vad_forw(dst1, T1, T2):
    fn = len(dst1)
    maxsilence = 8
    minlen = 5
    status = 0
    count = np.zeros(fn)
    silence = np.zeros(fn)
    xn = 0
    x1 = np.zeros(fn)
    x2 = np.zeros(fn)
    for n in range(1, fn):
        if status == 0 or status == 1:
            if dst1[n] > T2:
                x1[xn] = max(1, n - count[xn] - 1)
                status = 2
                silence[xn] = 0
                count[xn] += 1
            elif dst1[n] > T1:
                status = 1
                count[xn] += 1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        if status == 2:
            if dst1[n] > T1:
                count[xn] += 1
            else:
                silence[xn] += 1
                if silence[xn] < maxsilence:
                    count[xn] += 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        if status == 3:
            status = 0
            xn += 1
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    el = len(x1[:xn])
    if x1[el - 1] == 0:
        el -= 1
    if x2[el - 1] == 0:
        print('Error: Not find endding point!\n')
        x2[el] = fn
    SF = np.zeros(fn)
    NF = np.ones(fn)
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1
        NF[int(x1[i]):int(x2[i])] = 0
    voiceseg = findSegment(np.where(SF == 1)[0])
    vsl = len(voiceseg.keys())
    return voiceseg, vsl, SF, NF


def findSegment(express):
    """
    :param express:
    :return:
    """
    if express[0] == 0:
        voiceIndex = np.where(express)
    else:
        voiceIndex = express
    d_voice = np.where(np.diff(voiceIndex) > 1)[0]
    voiceseg = {}
    if len(d_voice) > 0:
        for i in range(len(d_voice) + 1):
            seg = {}
            if i == 0:
                st = voiceIndex[0]
                en = voiceIndex[d_voice[i]]
            elif i == len(d_voice):
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[-1]
            else:
                st = voiceIndex[d_voice[i - 1] + 1]
                en = voiceIndex[d_voice[i]]
            seg['start'] = st
            seg['end'] = en
            seg['duration'] = en - st + 1
            voiceseg[i] = seg
    return voiceseg

def STAc(x):
    """
    :param x:
    :return:
    """
    para = np.zeros(x.shape)
    fn = x.shape[1]
    for i in range(fn):
        R = np.correlate(x[:, i], x[:, i], 'valid')
        para[:, i] = R
    return para

def vad_corr(y, wnd, inc, NIS, th1, th2):
    x = enframe(y, wnd, inc)
    Ru = STAc(x.T)[0]
    Rum = Ru / np.max(Ru)
    thredth = np.max(Rum[:NIS])
    T1 = th1 * thredth
    T2 = th2 * thredth
    voiceseg, vsl, SF, NF = vad_forw(Rum, T1, T2)
    return voiceseg, vsl, SF, NF, Rum

def VAD(filepath):
    data, fs = soundBase(filepath).audioread()
    data -= np.mean(data)
    data /= np.max(data)
    IS = 0.25
    wlen = 200
    inc = 80
    N = len(data)
    time = [i / fs for i in range(N)]
    wnd = np.hamming(wlen)
    overlap = wlen - inc
    NIS = int((IS * fs - wlen) // inc + 1)
    thr1 = 0.995
    thr2 = 0.96
    voiceseg, vsl, SF, NF, Enm = vad_specEN(data, wnd, inc, NIS, thr1, thr2, fs)

    fn = len(SF)
    frameTime = FrameTimeC(fn, wlen, inc, fs)


    
    voiceseg_new = {}
    vsl_new = 0
    count = 1
    while(count):
        count = 0
        vsl_new = 0
        voiceseg_new = {}
        jump = 0
        for i in range(vsl):
            seg = {}
            if jump:
                jump=0
                continue
            if (i+1<vsl) and frameTime[voiceseg[i+1]['start']] - frameTime[voiceseg[i]['end']]<0.25:
                seg['start'] = voiceseg[i]['start']
                seg['end'] = voiceseg[i+1]['end']
                seg['duration'] = seg['end'] - seg['start'] + 1
                voiceseg_new[vsl_new] = seg
                jump = 1
                count += 1
                vsl_new += 1
            else:
                seg['start'] = voiceseg[i]['start']
                seg['end'] = voiceseg[i]['end']
                seg['duration'] = seg['end'] - seg['start'] + 1
                voiceseg_new[vsl_new] = seg
                vsl_new += 1

        voiceseg = voiceseg_new
        vsl = vsl_new
    return frameTime, voiceseg, vsl

