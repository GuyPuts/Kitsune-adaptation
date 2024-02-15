import heapq
import math
from collections import deque
from datetime import timedelta
from random import random
from tdigest import TDigest
import numpy as np


class incStat:
    def __init__(self, Lambda, ID, init_time=0, isTypeDiff=False, tcpFlags=False):  # timestamp is creation time
        self.ID = ID
        self.CF1 = 0  # linear sum
        self.CF2 = 0  # sum of squares
        self.w = 1e-20  # weight
        self.isTypeDiff = isTypeDiff
        self.Lambda = Lambda  # Decay Factor
        self.lastTimestamp = init_time
        self.cur_mean = np.nan
        self.cur_var = np.nan
        self.cur_std = np.nan
        self.decay = 1.0
        self.covs = [] # a list of incStat_covs (references) with relate to this incStat
        self.tcpPkts = 0
        self.flag_counts = {
            "FIN": 0.0,
            "SYN": 0.0,
            "RST": 0.0,
            "PSH": 0.0,
            "ACK": 0.0,
            "URG": 0.0,
            "ECE": 0.0,
            "CWR": 0.0
        }
        self.tcp_time_window = timedelta(minutes=5)
        self.tcpBuffer = []
        self.sqlinj_sentinel = 0.0
        self.xss_sentinel = 0.0
        self.ftp_threshold_size = 100
        self.ftp_threshold_packets = 5
        self.ftp_time_window = timedelta(minutes=5)
        self.ftp_packets = deque()
        self.ftp_recent_packets = deque()
        self.ssh_threshold_syn = 4
        self.ssh_time_window = timedelta(minutes=5)
        self.ssh_syn_flags = deque()
        self.width = 100
        self.depth = 5
        self.counters = [[0] * self.width for _ in range(self.depth)]
        self.min = False
        self.max = False
        self.tdigest = TDigest()

    def insert(self, v, t=0, tcpFlags=False, ftp=False, ssh=False, sqlinj=False, xss=False, median=False, minmax=False, quantiles=False):  # v is a scalar, t is v's arrival the timestamp
        if sqlinj:
            if sqlinj == 1.0:
                self.sqlinj_sentinel += 1.0
            return True
        if xss:
            if xss == 1.0:
                self.xss_sentinel += 1.0
            return True
        if ftp:
            self.ftp_packets.append((t, v))
            self.ftp_recent_packets.append((t, v))

            # Remove older packets outside the time window
            while self.ftp_recent_packets and t - self.ftp_recent_packets[0][0] > self.ftp_time_window.total_seconds():
                self.ftp_recent_packets.popleft()
            if not ssh:
                return True

        if ssh and not isinstance(tcpFlags, bool):
            flag_int = int(tcpFlags, 16)
            syn_flag = bool(flag_int & (1 << 1))  # Checking SYN flag

            if syn_flag:
                self.ssh_syn_flags.append((t, syn_flag))

                while self.ssh_syn_flags and t - self.ssh_syn_flags[0][0] > self.ssh_time_window:
                    self.ssh_syn_flags.popleft()
            return True

        if minmax:
            if not self.min:
                self.min = v
            if not self.max:
                self.max = v
            if v < self.min:
                self.min = v
            if v > self.max:
                self.max = v
            return True

        if self.isTypeDiff:
            dif = t - self.lastTimestamp
            if dif > 0:
                v = dif
            else:
                v = 0
        self.processDecay(t)

        if tcpFlags:
            flag_int = int(tcpFlags, 16)  # Convert hex string to integer
            flags = ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE", "CWR"]
            # pkt_dict = {
            #     'timestamp': t,
            #     'FIN': 0,
            #     'SYN': 0,
            #     'RST': 0,
            #     'PSH': 0,
            #     'ACK': 0,
            #     'URG': 0,
            #     'ECE': 0,
            #     'CWR': 0,
            # }
            for i, flag in enumerate(flags):
                if flag_int & (1 << i):  # Check if the flag is set
                    self.tcpPkts += 1
                    self.flag_counts[flag] += 1
                    # pkt_dict[flag] += 1

            # self.tcpBuffer.append(pkt_dict)
            # newbuffer = []
            # for dictionary in self.tcpBuffer:
            #     if not t - dictionary["timestamp"] > self.tcp_time_window.total_seconds():
            #         newbuffer.append(dictionary)
            # self.tcpBuffer = newbuffer
            # Process decay

            self.flag_counts["FIN"] = float(self.flag_counts["FIN"]) * self.decay
            self.flag_counts["SYN"] = float(self.flag_counts["SYN"]) * self.decay
            self.flag_counts["RST"] = float(self.flag_counts["RST"]) * self.decay
            self.flag_counts["PSH"] = float(self.flag_counts["PSH"]) * self.decay
            self.flag_counts["ACK"] = float(self.flag_counts["ACK"]) * self.decay
            self.flag_counts["URG"] = float(self.flag_counts["URG"]) * self.decay
            self.flag_counts["ECE"] = float(self.flag_counts["ECE"]) * self.decay
            self.flag_counts["CWR"] = float(self.flag_counts["CWR"]) * self.decay
            return True

        # update with v
        self.CF1 += v
        self.CF2 += math.pow(v, 2)
        self.w += 1
        self.cur_mean = np.nan  # force recalculation if called
        self.cur_var = np.nan
        self.cur_std = np.nan

        # update covs (if any)
        for cov in self.covs:
            cov.update_cov(self.ID, v, t)

        if median or quantiles:
            self.tdigest.update(v)
            for i in range(self.depth):
                hash_val = hash((i, v)) % self.width
                self.counters[i][hash_val] += 1

    def processDecay(self, timestamp):
        factor=1
        # check for decay
        timeDiff = timestamp - self.lastTimestamp
        if timeDiff > 0:
            factor = math.pow(2, (-self.Lambda * timeDiff))
            self.decay = factor
            self.CF1 = self.CF1 * factor
            self.CF2 = self.CF2 * factor
            self.w = self.w * factor
            self.lastTimestamp = timestamp
        return factor

    def weight(self):
        return self.w

    def mean(self):
        if math.isnan(self.cur_mean):  # calculate it only once when necessary
            self.cur_mean = self.CF1 / self.w
        return self.cur_mean

    def var(self):
        if math.isnan(self.cur_var):  # calculate it only once when necessary
            self.cur_var = abs(self.CF2 / self.w - math.pow(self.mean(), 2))
        return self.cur_var

    def std(self):
        if math.isnan(self.cur_std):  # calculate it only once when necessary
            self.cur_std = math.sqrt(self.var())
        return self.cur_std

    def cov(self,ID2):
        for cov in self.covs:
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                return cov.cov()
        return [np.nan]

    def pcc(self,ID2):
        for cov in self.covs:
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                return cov.pcc()
        return [np.nan]

    def cov_pcc(self,ID2):
        for cov in self.covs:
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                return cov.get_stats1()
        return [np.nan]*2

    def radius(self, other_incStats):  # the radius of a set of incStats
        A = self.var()**2
        for incS in other_incStats:
            A += incS.var()**2
        return math.sqrt(A)

    def magnitude(self, other_incStats):  # the magnitude of a set of incStats
        A = math.pow(self.mean(), 2)
        for incS in other_incStats:
            A += math.pow(incS.mean(), 2)
        return math.sqrt(A)

    #calculates and pulls all stats on this stream
    def allstats_1D(self, tcpFlags=False, tcpMean=False, ftp=False, ssh=False, sqlinj=False, xss=False, median=False, minmax=False, quantiles=False):
        self.cur_mean = self.CF1 / self.w
        self.cur_var = abs(self.CF2 / self.w - math.pow(self.cur_mean, 2))
        # Return mean of tcp flags
        if tcpFlags and not ssh:
            if self.tcpPkts > 0:
                if tcpMean:
                    return [flag / self.tcpPkts for flag in list(self.flag_counts.values())].append(self.tcpPkts)
                else:
                    return list(self.flag_counts.values())
            else:
                flags = [0, 0, 0, 0, 0, 0, 0, 0]
                return flags
            flags = [0, 0, 0, 0, 0, 0, 0, 0]
            if self.tcpBuffer:
                for dictionary in self.tcpBuffer:
                    flags[0] += dictionary["FIN"]
                    flags[1] += dictionary["SYN"]
                    flags[2] += dictionary["RST"]
                    flags[3] += dictionary["PSH"]
                    flags[4] += dictionary["ACK"]
                    flags[5] += dictionary["URG"]
                    flags[6] += dictionary["ECE"]
                    flags[7] += dictionary["CWR"]
                if tcpMean:
                    return [x / self.tcpPkts for x in flags]
            return flags
        elif ftp:
            count = sum(1 for packet in self.ftp_recent_packets if packet[1] < self.ftp_threshold_size)
            return float(count > self.ftp_threshold_packets)
        elif ssh:
            count_syn_flags = sum(1 for _, flag in self.ssh_syn_flags if flag)
            return float(count_syn_flags > self.ssh_threshold_syn)
        elif sqlinj:
            return self.sqlinj_sentinel
        elif xss:
            return self.xss_sentinel
        elif minmax:
            minmax = abs(self.max - self.min)
            return minmax
        elif median:
            median = self.tdigest.percentile(50)
            return [self.w, self.cur_mean, self.cur_var, median]
            heap = []
            for i in range(self.width):
                freq = 0
                for j in range(self.depth):
                    freq += self.counters[j][i]
                freq *= self.decay ** i  # Apply decay
                heapq.heappush(heap, freq)

            median_idx = len(heap) // 2
            median = heapq.nsmallest(median_idx + 1, heap)[-1]
            return [self.w, self.cur_mean, self.cur_var, median]
        elif quantiles:
            values = []
            for quantile in quantiles:
                values.append(self.tdigest.percentile(quantile))
            return values
        return [self.w, self.cur_mean, self.cur_var]

    #calculates and pulls all stats on this stream, and stats shared with the indicated stream
    def allstats_2D(self, ID2):
        stats1D = self.allstats_1D()
        # Find cov component
        stats2D = [np.nan] * 4
        for cov in self.covs:
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                stats2D = cov.get_stats2()
                break
        return stats1D + stats2D

    def getHeaders_1D(self, suffix=True):
        if self.ID is None:
            s0=""
        else:
            s0 = "_0"
        if suffix:
            s0 = "_"+self.ID
        headers = ["weight"+s0, "mean"+s0, "std"+s0]
        return headers

    def getHeaders_2D(self, ID2, suffix=True):
        hdrs1D = self.getHeaders_1D(suffix)
        if self.ID is None:
            s0=""
            s1=""
        else:
            s0 = "_0"
            s1 = "_1"
        if suffix:
            s0 = "_"+self.ID
            s1 = "_" + ID2
        hdrs2D = ["radius_" + s0 + "_" + s1, "magnitude_" + s0 + "_" + s1, "covariance_" + s0 + "_" + s1,
                   "pcc_" + s0 + "_" + s1]
        return hdrs1D+hdrs2D


#like incStat, but maintains stats between two streams
class incStat_cov:
    def __init__(self, incS1, incS2, init_time = 0):
        # store references tot he streams' incStats
        self.incStats = [incS1,incS2]
        self.lastRes = [0,0]
        # init extrapolators
        #self.EXs = [extrapolator(),extrapolator()]

        # init sum product residuals
        self.CF3 = 0 # sum of residule products (A-uA)(B-uB)
        self.w3 = 1e-20
        self.lastTimestamp_cf3 = init_time

    #other_incS_decay is the decay factor of the other incstat
    # ID: the stream ID which produced (v,t)
    def update_cov(self, ID, v, t):  # it is assumes that incStat "ID" has ALREADY been updated with (t,v) [this si performed automatically in method incStat.insert()]
        # find incStat
        if ID == self.incStats[0].ID:
            inc = 0
        elif ID == self.incStats[1].ID:
            inc = 1
        else:
            print("update_cov ID error")
            return ## error

        # Decay other incStat
        self.incStats[not(inc)].processDecay(t)

        # Decay residules
        self.processDecay(t,inc)

        # Update extrapolator for current stream
        #self.EXs[inc].insert(t,v)

        # Extrapolate other stream
        #v_other = self.EXs[not(inc)].predict(t)

        # Compute and update residule
        res = (v - self.incStats[inc].mean())
        resid = (v - self.incStats[inc].mean()) * self.lastRes[not(inc)]
        self.CF3 += resid
        self.w3 += 1
        self.lastRes[inc] = res

    def processDecay(self,t,micro_inc_indx):
        factor = 1
        # check for decay cf3
        timeDiffs_cf3 = t - self.lastTimestamp_cf3
        if timeDiffs_cf3 > 0:
            factor = math.pow(2, (-(self.incStats[micro_inc_indx].Lambda) * timeDiffs_cf3))
            self.CF3 *= factor
            self.w3 *= factor
            self.lastTimestamp_cf3 = t
            self.lastRes[micro_inc_indx] *= factor
        return factor

    #todo: add W3 for cf3

    #covariance approximation
    def cov(self):
        return self.CF3 / self.w3

    # Pearson corl. coef
    def pcc(self):
        ss = self.incStats[0].std() * self.incStats[1].std()
        if ss != 0:
            return self.cov() / ss
        else:
            return 0

    # calculates and pulls all correlative stats
    def get_stats1(self):
        return [self.cov(), self.pcc()]

    # calculates and pulls all correlative stats AND 2D stats from both streams (incStat)
    def get_stats2(self):
        return [self.incStats[0].radius([self.incStats[1]]),self.incStats[0].magnitude([self.incStats[1]]),self.cov(), self.pcc()]

    # calculates and pulls all correlative stats AND 2D stats AND the regular stats from both streams (incStat)
    def get_stats3(self):
        return [self.incStats[0].w,self.incStats[0].mean(),self.incStats[0].std(),self.incStats[1].w,self.incStats[1].mean(),self.incStats[1].std(),self.cov(), self.pcc()]

    # calculates and pulls all correlative stats AND the regular stats from both incStats AND 2D stats
    def get_stats4(self):
        return [self.incStats[0].w,self.incStats[0].mean(),self.incStats[0].std(),self.incStats[1].w,self.incStats[1].mean(),self.incStats[1].std(), self.incStats[0].radius([self.incStats[1]]),self.incStats[0].magnitude([self.incStats[1]]),self.cov(), self.pcc()]

    def getHeaders(self,ver,suffix=True): #ver = {1,2,3,4}
        headers = []
        s0 = "0"
        s1 = "1"
        if suffix:
            s0 = self.incStats[0].ID
            s1 = self.incStats[1].ID

        if ver == 1:
            headers = ["covariance_"+s0+"_"+s1, "pcc_"+s0+"_"+s1]
        if ver == 2:
            headers = ["radius_"+s0+"_"+s1, "magnitude_"+s0+"_"+s1, "covariance_"+s0+"_"+s1, "pcc_"+s0+"_"+s1]
        if ver == 3:
            headers = ["weight_"+s0, "mean_"+s0, "std_"+s0,"weight_"+s1, "mean_"+s1, "std_"+s1, "covariance_"+s0+"_"+s1, "pcc_"+s0+"_"+s1]
        if ver == 4:
            headers = ["weight_" + s0, "mean_" + s0, "std_" + s0, "covariance_" + s0 + "_" + s1, "pcc_" + s0 + "_" + s1]
        if ver == 5:
            headers = ["weight_"+s0, "mean_"+s0, "std_"+s0,"weight_"+s1, "mean_"+s1, "std_"+s1, "radius_"+s0+"_"+s1, "magnitude_"+s0+"_"+s1, "covariance_"+s0+"_"+s1, "pcc_"+s0+"_"+s1]
        return headers


class incStatDB:
    # default_lambda: use this as the lambda for all streams. If not specified, then you must supply a Lambda with every query.
    def __init__(self,limit=np.Inf,default_lambda=np.nan):
        self.HT = dict()
        self.limit = limit
        self.df_lambda = default_lambda

    def get_lambda(self,Lambda):
        if not np.isnan(self.df_lambda):
            Lambda = self.df_lambda
        return Lambda

    # Registers a new stream. init_time: init lastTimestamp of the incStat
    def register(self,ID,Lambda=1,init_time=0,isTypeDiff=False):
        #Default Lambda?
        Lambda = self.get_lambda(Lambda)
        #Retrieve incStat
        key = ID+"_"+str(Lambda)
        incS = self.HT.get(key)
        if incS is None: #does not already exist
            if len(self.HT) + 1 > self.limit:
                raise LookupError(
                    'Adding Entry:\n' + key + '\nwould exceed incStatHT 1D limit of ' + str(
                        self.limit) + '.\nObservation Rejected.')
            incS = incStat(Lambda, ID, init_time, isTypeDiff)
            self.HT[key] = incS #add new entry
        return incS

    # Registers covariance tracking for two streams, registers missing streams
    def register_cov(self,ID1,ID2,Lambda=1,init_time=0,isTypeDiff=False):
        #Default Lambda?
        Lambda = self.get_lambda(Lambda)

        # Lookup both streams
        incS1 = self.register(ID1,Lambda,init_time,isTypeDiff)
        incS2 = self.register(ID2,Lambda,init_time,isTypeDiff)

        #check for pre-exiting link
        for cov in incS1.covs:
            if cov.incStats[0].ID == ID2 or cov.incStats[1].ID == ID2:
                return cov #there is a pre-exiting link

        # Link incStats
        inc_cov = incStat_cov(incS1,incS2,init_time)
        incS1.covs.append(inc_cov)
        incS2.covs.append(inc_cov)
        return inc_cov

    # updates/registers stream
    def update(self,ID,t,v,Lambda=1,isTypeDiff=False,tcpFlags=False,ftp=False,ssh=False,sqlinj=False,sqltype=False,xss=False,xsstype=False,median=False,quantiles=False,minmax=False):
        if median:
            incS = self.register(f"median_{ID}",Lambda,t,isTypeDiff)
        elif tcpFlags:
            incS = self.register(f"tcp_{ID}",Lambda,t,isTypeDiff)
        elif ftp:
            incS = self.register(f"ftp_{ID}", Lambda, t, isTypeDiff)
        elif ssh:
            incS = self.register(f"ssh_{ID}", Lambda, t, isTypeDiff)
        elif sqlinj:
            incS = self.register(f"sqlinj_{sqltype}_{ID}", Lambda, t, isTypeDiff)
        elif xss:
            incS = self.register(f"xss_{xsstype}_{ID}", Lambda, t, isTypeDiff)
        elif minmax:
            incS = self.register(f"minmax_{ID}", Lambda, t, isTypeDiff)
        else:
            incS = self.register(ID, Lambda, t, isTypeDiff)
        incS.insert(v,t,tcpFlags=tcpFlags,ftp=ftp,ssh=ssh,sqlinj=sqlinj,xss=xss,median=median,quantiles=quantiles,minmax=minmax)
        return incS

    # Pulls current stats from the given ID
    def get_1D_Stats(self,ID,Lambda=1): #weight, mean, std
        #Default Lambda?
        Lambda = self.get_lambda(Lambda)

        #Get incStat
        incS = self.HT.get(ID+"_"+str(Lambda))
        if incS is None:  # does not already exist
            return [np.na]*3
        else:
            return incS.allstats_1D()

    # Pulls current correlational stats from the given IDs
    def get_2D_Stats(self, ID1, ID2, Lambda=1): #cov, pcc
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)

        # Get incStat
        incS1 = self.HT.get(ID1 + "_" + str(Lambda))
        if incS1 is None:  # does not exist
            return [np.na]*2

        # find relevant cov entry
        return incS1.cov_pcc(ID2)

    # Pulls all correlational stats registered with the given ID
    # returns tuple [0]: stats-covs&pccs, [2]: IDs
    def get_all_2D_Stats(self, ID, Lambda=1):  # cov, pcc
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)

        # Get incStat
        incS1 = self.HT.get(ID + "_" + str(Lambda))
        if incS1 is None:  # does not exist
            return ([],[])

        # find relevant cov entry
        stats = []
        IDs = []
        for cov in incS1.covs:
            stats.append(cov.get_stats1())
            IDs.append([cov.incStats[0].ID,cov.incStats[1].ID])
        return stats,IDs

    # Pulls current multidimensional stats from the given IDs
    def get_nD_Stats(self,IDs,Lambda=1): #radius, magnitude (IDs is a list)
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)

        # Get incStats
        incStats = []
        for ID in IDs:
            incS = self.HT.get(ID + "_" + str(Lambda))
            if incS is not None:  #exists
                incStats.append(incS)

        # Compute stats
        rad = 0 #radius
        mag = 0 #magnitude
        for incS in incStats:
            rad += incS.var()
            mag += incS.mean()**2

        return [np.sqrt(rad),np.sqrt(mag)]

    # Updates and then pulls current 1D stats from the given ID. Automatically registers previously unknown stream IDs
    def update_get_1D_Stats(self, ID,t,v,Lambda=1,isTypeDiff=False, tcpFlags=False, tcpMean=False, ftp=False, ssh=False, sqlinj=False, sqltype=False, xss=False, xsstype=False, median=False, minmax=False, quantiles=False):  # weight, mean, std
        incS = self.update(ID,t,v,Lambda,isTypeDiff, tcpFlags=tcpFlags, ftp=ftp, ssh=ssh, sqlinj=sqlinj, sqltype=sqltype, xss=xss, xsstype=xsstype, median=median, minmax=minmax, quantiles=quantiles)
        stats = incS.allstats_1D(tcpFlags, tcpMean, ftp, ssh, sqlinj, xss, median, minmax, quantiles)
        return stats


    # Updates and then pulls current correlative stats between the given IDs. Automatically registers previously unknown stream IDs, and cov tracking
    #Note: AfterImage does not currently support Diff Type streams for correlational statistics.
    def update_get_2D_Stats(self, ID1,ID2,t1,v1,Lambda=1,level=1):  #level=  1:cov,pcc  2:radius,magnitude,cov,pcc
        #retrieve/add cov tracker
        inc_cov = self.register_cov(ID1, ID2, Lambda,  t1)
        # Update cov tracker
        inc_cov.update_cov(ID1,v1,t1)
        if level == 1:
            return inc_cov.get_stats1()
        else:
            return inc_cov.get_stats2()

    # Updates and then pulls current 1D and 2D stats from the given IDs. Automatically registers previously unknown stream IDs
    def update_get_1D2D_Stats(self, ID1,ID2,t1,v1,Lambda=1,median=False,quantiles=False):  # weight, mean, std
        if quantiles:
            return self.update_get_1D_Stats(ID1, t1, v1, Lambda, median=median, quantiles=quantiles)
        return self.update_get_1D_Stats(ID1,t1,v1,Lambda,median=median,quantiles=quantiles) + self.update_get_2D_Stats(ID1,ID2,t1,v1,Lambda,level=2)

    def getHeaders_1D(self,Lambda=1,ID=None):
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)
        hdrs = incStat(Lambda,ID).getHeaders_1D(suffix=False)
        return [str(Lambda)+"_"+s for s in hdrs]

    def getHeaders_2D(self,Lambda=1,IDs=None, ver=1): #IDs is a 2-element list or tuple
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)
        if IDs is None:
            IDs = [0,1]
        hdrs = incStat_cov(incStat(Lambda,IDs[0]),incStat(Lambda,IDs[0]),Lambda).getHeaders(ver,suffix=False)
        return [str(Lambda)+"_"+s for s in hdrs]

    def getHeaders_1D2D(self,Lambda=1,IDs=None, ver=1):
        # Default Lambda?
        Lambda = self.get_lambda(Lambda)
        if IDs is None:
            IDs = [0,1]
        hdrs1D = self.getHeaders_1D(Lambda,IDs[0])
        hdrs2D = self.getHeaders_2D(Lambda,IDs, ver)
        return hdrs1D + hdrs2D

    def getHeaders_nD(self,Lambda=1,IDs=[]): #IDs is a n-element list or tuple
        # Default Lambda?
        ID = ":"
        for s in IDs:
            ID += "_"+s
        Lambda = self.get_lambda(Lambda)
        hdrs = ["radius"+ID, "magnitude"+ID]
        return [str(Lambda)+"_"+s for s in hdrs]


    #cleans out records that have a weight less than the cutoff.
    #returns number or removed records.
    def cleanOutOldRecords(self,cutoffWeight,curTime):
        n = 0
        dump = sorted(self.HT.items(), key=lambda tup: tup[1][0].getMaxW(curTime))
        for entry in dump:
            entry[1][0].processDecay(curTime)
            W = entry[1][0].w
            if W <= cutoffWeight:
                key = entry[0]
                del entry[1][0]
                del self.HT[key]
                n=n+1
            elif W > cutoffWeight:
                break
        return n
