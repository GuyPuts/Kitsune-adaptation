import re

import numpy as np
## Prep AfterImage cython package
import os
import subprocess
import pyximport
pyximport.install()
import AfterImage as af
#import AfterImage_NDSS as af

#
# MIT License
#
# Copyright (c) 2018 Yisroel mirsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class netStat:
    #Datastructure for efficent network stat queries
    # HostLimit: no more that this many Host identifiers will be tracked
    # HostSimplexLimit: no more that this many outgoing channels from each host will be tracked (purged periodically)
    # Lambdas: a list of 'window sizes' (decay factors) to track for each stream. nan resolved to default [5,3,1,.1,.01]
    def __init__(self, Lambdas = np.nan, HostLimit=255,HostSimplexLimit=1000):
        #Lambdas
        if np.isnan(Lambdas):
            self.Lambdas = [5,3,1,.1,.01]
        else:
            self.Lambdas = Lambdas

        #HT Limits
        self.HostLimit = HostLimit
        self.SessionLimit = HostSimplexLimit*self.HostLimit*self.HostLimit #*2 since each dual creates 2 entries in memory
        self.MAC_HostLimit = self.HostLimit*10

        #HTs
        self.HT_jit = af.incStatDB(limit=self.HostLimit*self.HostLimit)#H-H Jitter Stats
        self.HT_MI = af.incStatDB(limit=self.MAC_HostLimit)#MAC-IP relationships
        self.HT_H = af.incStatDB(limit=self.HostLimit) #Source Host BW Stats
        self.HT_Hp = af.incStatDB(limit=self.SessionLimit)#Source Host BW Stats

        self.HT_MI_jit = af.incStatDB(limit=self.HostLimit)#Source Host Jitter Stats
        self.HT_Hp_jit = af.incStatDB(limit=self.SessionLimit*self.SessionLimit)#Socket-Socket Jitter Stats

        self.DT_MI = af.incStatDB(limit=self.MAC_HostLimit)#MAC-IP relationships (DST)
        self.DT_MI_jit = af.incStatDB(limit=self.HostLimit)  # Destination Host Jitter Stats

        #Flags
        self.HT_MI_FLAG_MEAN = af.incStatDB(limit=self.MAC_HostLimit) # Flags of SRC IP
        self.HT_H_FLAG_MEAN = af.incStatDB(limit=self.HostLimit)  # Flags of Channel
        self.HT_Hp_FLAG_MEAN = af.incStatDB(limit=self.SessionLimit)  # Flags of Socket
        self.DT_MI_FLAG_MEAN = af.incStatDB(limit=self.MAC_HostLimit) # Flas of DST IP
        self.HT_MI_FLAG_COUNT = af.incStatDB(limit=self.MAC_HostLimit)  # Flags of SRC IP
        self.HT_H_FLAG_COUNT = af.incStatDB(limit=self.HostLimit)  # Flags of Channel
        self.HT_Hp_FLAG_COUNT = af.incStatDB(limit=self.SessionLimit)  # Flags of Socket
        self.DT_MI_FLAG_COUNT = af.incStatDB(limit=self.MAC_HostLimit)  # Flas of DST IP

        #Quantiles
        self.HT_MI_QUANT = af.incStatDB(limit=self.MAC_HostLimit)  # Flags of SRC IP
        self.HT_H_QUANT = af.incStatDB(limit=self.HostLimit)  # Flags of Channel
        self.HT_Hp_QUANT = af.incStatDB(limit=self.SessionLimit)  # Flags of Socket
        self.DT_MI_QUANT = af.incStatDB(limit=self.MAC_HostLimit)  # Flas of DST IP


    def findDirection(self,IPtype,srcIP,dstIP,eth_src,eth_dst): #cpp: this is all given to you in the direction string of the instance (NO NEED FOR THIS FUNCTION)
        if IPtype==0: #is IPv4
            lstP = srcIP.rfind('.')
            src_subnet = srcIP[0:lstP:]
            lstP = dstIP.rfind('.')
            dst_subnet = dstIP[0:lstP:]
        elif IPtype==1: #is IPv6
            src_subnet = srcIP[0:round(len(srcIP)/2):]
            dst_subnet = dstIP[0:round(len(dstIP)/2):]
        else: #no Network layer, use MACs
            src_subnet = eth_src
            dst_subnet = eth_dst

        return src_subnet, dst_subnet

    def updateGetStats(self, IPtype, srcMAC,dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, datagramSize, timestamp, tcpFlags=False, payload = 0, ftp=False, ssh=False, sqlinj=False, xss=False, median=False, minmax=False, kind=1):
        if kind == 1:
            return "error"
        if kind == 2:
            return self.updateGetStatsFirstHalfSecondPart(IPtype, srcMAC,dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, datagramSize, timestamp, tcpFlags=tcpFlags)
        if kind == 3:
            return self.updateGetStatsSecondHalf(IPtype, srcMAC,dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, datagramSize, timestamp, tcpFlags=tcpFlags)
    #     # Host BW: Stats on the srcIP's general Sender Statistics
    #     # Hstat = np.zeros((3*len(self.Lambdas,)))
    #     # for i in range(len(self.Lambdas)):
    #     #     Hstat[(i*3):((i+1)*3)] = self.HT_H.update_get_1D_Stats(srcIP, timestamp, datagramSize, self.Lambdas[i])
    #
    #
    #     #MAC.IP: Stats on src MAC-IP relationships
    #     if median:
    #         MIstat =  np.zeros(4*len(self.Lambdas,))
    #         for i in range(len(self.Lambdas)):
    #             MIstat[(i*4):((i+1)*4)] = self.HT_MI.update_get_1D_Stats(srcMAC+srcIP, timestamp, datagramSize, self.Lambdas[i], median=True)
    #     else:
    #         MIstat = np.zeros((3 * len(self.Lambdas, )))
    #         for i in range(len(self.Lambdas)):
    #             MIstat[(i * 3):((i + 1) * 3)] = self.HT_MI.update_get_1D_Stats(srcMAC + srcIP, timestamp, datagramSize,
    #                                                                            self.Lambdas[i])
    #
    #     # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
    #     HHstat =  np.zeros((7*len(self.Lambdas,)))
    #     for i in range(len(self.Lambdas)):
    #         HHstat[(i*7):((i+1)*7)] = self.HT_H.update_get_1D2D_Stats(srcIP, dstIP,timestamp,datagramSize,self.Lambdas[i])
    #
    #     # Host-Host Jitter:
    #     HHstat_jit =  np.zeros((3*len(self.Lambdas,)))
    #     for i in range(len(self.Lambdas)):
    #         HHstat_jit[(i*3):((i+1)*3)] = self.HT_jit.update_get_1D_Stats(srcIP+dstIP, timestamp, 0, self.Lambdas[i],isTypeDiff=True)
    #
    #     # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
    #     HpHpstat =  np.zeros((7*len(self.Lambdas,)))
    #     if srcProtocol == 'arp':
    #         for i in range(len(self.Lambdas)):
    #             HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.update_get_1D2D_Stats(srcMAC, dstMAC, timestamp, datagramSize, self.Lambdas[i])
    #     else:  # some other protocol (e.g. TCP/UDP)
    #         for i in range(len(self.Lambdas)):
    #             HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.update_get_1D2D_Stats(srcIP + srcProtocol, dstIP + dstProtocol, timestamp, datagramSize, self.Lambdas[i])
    #
    #     if not tcpFlags:
    #         tcpstat = np.zeros(8)
    #     if tcpFlags and tcpFlags == "":
    #         tcpstat = np.zeros(8)
    #     if tcpFlags and tcpFlags != "":
    #         # MAC.IP: Stats on src MAC-IP relationships
    #         tcpstat = np.zeros(8)
    #         tcpstat[0:9] = self.HT_MI.update_get_1D_Stats(srcIP+dstIP, timestamp, datagramSize, self.Lambdas[i], tcpFlags=tcpFlags)
    #
    #     ftpstat = np.zeros(1)
    #     ftpPorts = ['21']
    #     if ftp and dstProtocol in ftpPorts:
    #         ftpstat[0] = self.HT_MI.update_get_1D_Stats(srcMAC + srcIP, timestamp, datagramSize,
    #                                                     self.Lambdas[i], ftp=True)
    #     sshstat = np.zeros(1)
    #     sshPorts = ['22']
    #     if ssh and dstProtocol in sshPorts:
    #         sshstat[0] = self.HT_MI.update_get_1D_Stats(srcMAC + srcIP, timestamp, datagramSize,
    #                                                     self.Lambdas[i], tcpFlags=tcpFlags, ssh=True)
    #     sqlinjstat = np.zeros(1)
    #     if sqlinj:
    #         # Check if the test string matches the regex pattern
    #         pattern_str = r'\w*%27((%61|a|%41)(%6E|n|%4E)(%64|d|%44))|((%75|u|%55)(%6E|n|%4E)(%69|i|%49)(%6F|o|%4F)(%6E|n|%4E))|((%73|s|%53)(%65|e|%45)(%6C|l|%4C)(%65|e|%45)(%63|c|%43)(%74|t|%54))'
    #         pattern = re.compile(pattern_str, re.IGNORECASE)
    #         match = pattern.search(sqlinj)
    #         if match:
    #             sqlinjstat[0] = self.HT_MI.update_get_1D_Stats(srcMAC + srcIP, timestamp, datagramSize,
    #                                                         self.Lambdas[i], sqlinj=1.0)
    #         else:
    #             sqlinjstat[0] = self.HT_MI.update_get_1D_Stats(srcMAC + srcIP, timestamp, datagramSize,
    #                                                            self.Lambdas[i], sqlinj=0.1)
    #     else:
    #         sqlinjstat[0] = self.HT_MI.update_get_1D_Stats(srcMAC + srcIP, timestamp, datagramSize,
    #                                                        self.Lambdas[i], sqlinj=0.1)
    #     xssstat = np.zeros(1)
    #     if xss:
    #         # Check if the test string matches the regex pattern
    #         pattern_str = r'\s*(?:%3C|<)\s*(?:%73|s|%53)\s*(?:%63|c|%43)\s*(?:%72|r|%52)\s*(?:%69|i|%49)\s*(?:%70|p|%50)\s*(?:%74|t|%54)|console\.log'
    #
    #         pattern = re.compile(pattern_str, re.IGNORECASE)
    #         match = pattern.search(xss)
    #         if match:
    #             xssstat[0] = self.HT_MI.update_get_1D_Stats(srcMAC + srcIP, timestamp, datagramSize,
    #                                                            self.Lambdas[i], xss=1.0)
    #         else:
    #             xssstat[0] = self.HT_MI.update_get_1D_Stats(srcMAC + srcIP, timestamp, datagramSize,
    #                                                            self.Lambdas[i], xss=0.1)
    #     else:
    #         xssstat[0] = self.HT_MI.update_get_1D_Stats(srcMAC + srcIP, timestamp, datagramSize,
    #                                                        self.Lambdas[i], xss=0.1)
    #     minmaxstat = np.zeros(1)
    #     if minmax:
    #         minmaxstat[0] = self.HT_MI.update_get_1D_Stats(srcMAC + srcIP, timestamp, datagramSize,
    #                                                        self.Lambdas[i], minmax=True)
    #
    #     return np.concatenate((MIstat, HHstat, HHstat_jit, HpHpstat, tcpstat, ftpstat, sshstat, sqlinjstat, xssstat, minmaxstat))  # concatenation of stats into one stat vector
    #     #return np.concatenate((MIstat, HHstat, HHstat_jit, HpHpstat))  # concatenation of stats into one stat vector

    def updateGetStatsFirstHalfSecondPart(self, IPtype, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, datagramSize, timestamp, tcpFlags=False):
        # Host BW: Stats on the srcIP's general Sender Statistics
        # Hstat = np.zeros((3*len(self.Lambdas,)))
        # for i in range(len(self.Lambdas)):
        #     Hstat[(i*3):((i+1)*3)] = self.HT_H.update_get_1D_Stats(srcIP, timestamp, datagramSize, self.Lambdas[i])

        # MAC.IP: Stats on src MAC-IP relationships
        onedimensionalfeaturecount = 3
        twodimensionalfeaturecount = 7
        median=False
        # MIstat = np.zeros((3 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     MIstat[(i * 3):((i + 1) * 3)] = self.HT_MI.update_get_1D_Stats(srcMAC + srcIP, timestamp, datagramSize,
        #                                                                    self.Lambdas[i],median=median)
        # if len(MIstat) != len(self.Lambdas) * 3:
        #     print(self.Lambdas*3)
        #     print('issue in MIstat')
        #
        # # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        # HHstat = np.zeros((7 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     HHstat[(i * 7):((i + 1) * 7)] = self.HT_H.update_get_1D2D_Stats(srcIP, dstIP, timestamp, datagramSize,
        #                                                                     self.Lambdas[i],median=median)
        # if len(HHstat) != len(self.Lambdas) * 7:
        #     print('issue in HHstat')
        # Host-Host Jitter:
        # HHstat_jit = np.zeros((4 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     HHstat_jit[(i * 4):((i + 1) * 4)] = self.HT_jit.update_get_1D_Stats(srcIP + dstIP, timestamp, 0,
        #                                                                         self.Lambdas[i], isTypeDiff=True,quantiles=[50])
        # if len(HHstat_jit) != len(self.Lambdas) * 4:
        #     print('issue in HHstat_jit')
        # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        HpHpstat = np.zeros((7 * len(self.Lambdas, )))
        if srcProtocol == 'arp':
            for i in range(len(self.Lambdas)):
                HpHpstat[(i * 7):((i + 1) * 7)] = self.HT_Hp.update_get_1D2D_Stats(srcMAC, dstMAC, timestamp,
                                                                                   datagramSize, self.Lambdas[i],median=median)
        else:  # some other protocol (e.g. TCP/UDP)
            for i in range(len(self.Lambdas)):
                HpHpstat[(i * 7):((i + 1) * 7)] = self.HT_Hp.update_get_1D2D_Stats(srcIP + srcProtocol,
                                                                                   dstIP + dstProtocol, timestamp,
                                                                                   datagramSize, self.Lambdas[i],median=median)
        if len(HpHpstat) != len(self.Lambdas) * 7:
            print('issue in HpHpstat')
        # New code starts here
        # HtMiJitstat = np.zeros((4 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     HtMiJitstat[(i * 4):((i + 1) * 4)] = self.HT_MI_jit.update_get_1D_Stats(srcIP, timestamp, datagramSize,
        #                                                                    self.Lambdas[i], isTypeDiff=True,quantiles=[50])
        # if len(HtMiJitstat) != len(self.Lambdas) * 4:
        #     print('issue in HtMiJitstat')
        # HtHpJitstat = np.zeros((4* len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     HtHpJitstat[(i * 4):((i + 1) * 4)] = self.HT_Hp_jit.update_get_1D_Stats(srcIP+srcProtocol+dstIP+dstProtocol, timestamp, datagramSize,
        #                                                                    self.Lambdas[i], isTypeDiff=True,quantiles=[50])
        # if len(HtHpJitstat) != len(self.Lambdas) * 4:
        #     print('issue in HtHpJitstat')
        # # DST stats
        # DT_MIstat = np.zeros((onedimensionalfeaturecount * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     DT_MIstat[(i * onedimensionalfeaturecount):((i + 1) * onedimensionalfeaturecount)] = self.DT_MI.update_get_1D_Stats(dstIP, timestamp, datagramSize,
        #                                                                    self.Lambdas[i],median=median)
        # if len(DT_MIstat) != len(self.Lambdas) * 3:
        #     print('issue in DT_MIstat')
        # # DST Jitter
        # DtMiJitstat = np.zeros((4 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     DtMiJitstat[(i * 4):((i + 1) * 4)] = self.DT_MI_jit.update_get_1D_Stats(dstIP, timestamp, datagramSize,
        #                                                                             self.Lambdas[i], isTypeDiff=True,quantiles=[50])
        # if len(DtMiJitstat) != len(self.Lambdas) * 4:
        #     print('issue in DtMiJitstat')
        # Flag means
        # MI_flagstat_mean = np.zeros((9 * len(self.Lambdas, )))
        # if tcpFlags and tcpFlags != "":
        #     for i in range(len(self.Lambdas)):
        #         MI_flagstat_mean[(i * 9):((i + 1) * 9)] = self.HT_MI_FLAG_MEAN.update_get_1D_Stats(srcIP, timestamp, datagramSize, self.Lambdas[i], tcpFlags=tcpFlags, tcpMean=True)
        #
        # if len(MI_flagstat_mean) != len(self.Lambdas) * 9:
        #     print('issue in MI_flagstat_mean')
        # H_flagstat_mean = np.zeros((9 * len(self.Lambdas, )))
        # if tcpFlags and tcpFlags != "":
        #     for i in range(len(self.Lambdas)):
        #         H_flagstat_mean[(i * 9):((i + 1) * 9)] = self.HT_H_FLAG_MEAN.update_get_1D_Stats(srcIP+dstIP, timestamp, datagramSize,
        #                                                                                  self.Lambdas[i], tcpFlags=tcpFlags, tcpMean=True)
        # if len(H_flagstat_mean) != len(self.Lambdas) * 9:
        #     print('issue in H_flagstat_mean')
        # HT_Hp_flagstat_mean = np.zeros((9 * len(self.Lambdas, )))
        # if tcpFlags and tcpFlags != "":
        #     for i in range(len(self.Lambdas)):
        #         HT_Hp_flagstat_mean[(i * 9):((i + 1) * 9)] = self.HT_Hp_FLAG_MEAN.update_get_1D_Stats(srcIP+srcProtocol+dstIP+dstProtocol, timestamp, datagramSize,
        #                                                                                  self.Lambdas[i], tcpFlags=tcpFlags, tcpMean=True)
        # if len(HT_Hp_flagstat_mean) != len(self.Lambdas) * 9:
        #     print('issue in HT_Hp_flagstat_mean')
        # DT_MI_flagstat_mean = np.zeros((9 * len(self.Lambdas, )))
        # if tcpFlags and tcpFlags != "":
        #     for i in range(len(self.Lambdas)):
        #         DT_MI_flagstat_mean[(i * 9):((i + 1) * 9)] = self.DT_MI_FLAG_MEAN.update_get_1D_Stats(dstIP, timestamp, datagramSize,
        #                                                                                     self.Lambdas[i],
        #                                                                                     tcpFlags=tcpFlags, tcpMean=True)
        # if len(DT_MI_flagstat_mean) != len(self.Lambdas) * 9:
        #     print('issue in DT_MI_flagstat_mean')
        # Flag counts
        # MI_flagstat_count = np.zeros((8 * len(self.Lambdas, )))
        # if tcpFlags and tcpFlags != "":
        #     for i in range(len(self.Lambdas)):
        #         MI_flagstat_count[(i * 8):((i + 1) * 8)] = self.HT_MI_FLAG_COUNT.update_get_1D_Stats(srcIP, timestamp,
        #                                                                                            datagramSize,
        #                                                                                            self.Lambdas[i],
        #                                                                                            tcpFlags=tcpFlags,
        #                                                                                            tcpMean=False)
        # H_flagstat_count = np.zeros((8 * len(self.Lambdas, )))
        # if tcpFlags and tcpFlags != "":
        #     for i in range(len(self.Lambdas)):
        #         H_flagstat_count[(i * 8):((i + 1) * 8)] = self.HT_H_FLAG_COUNT.update_get_1D_Stats(srcIP + dstIP,
        #                                                                                          timestamp,
        #                                                                                          datagramSize,
        #                                                                                          self.Lambdas[i],
        #                                                                                          tcpFlags=tcpFlags,
        #                                                                                          tcpMean=False)
        # HT_Hp_flagstat_count = np.zeros((8 * len(self.Lambdas, )))
        # if tcpFlags and tcpFlags != "":
        #     for i in range(len(self.Lambdas)):
        #         HT_Hp_flagstat_count[(i * 8):((i + 1) * 8)] = self.HT_Hp_FLAG_COUNT.update_get_1D_Stats(
        #             srcIP + srcProtocol + dstIP + dstProtocol, timestamp, datagramSize,
        #             self.Lambdas[i], tcpFlags=tcpFlags, tcpMean=False)
        # DT_MI_flagstat_count = np.zeros((8 * len(self.Lambdas, )))
        # if tcpFlags and tcpFlags != "":
        #     for i in range(len(self.Lambdas)):
        #         DT_MI_flagstat_count[(i * 8):((i + 1) * 8)] = self.DT_MI_FLAG_COUNT.update_get_1D_Stats(dstIP, timestamp,
        #                                                                                               datagramSize,
        #                                                                                               self.Lambdas[i],
        #                                                                                               tcpFlags=tcpFlags,
        #                                                                                               tcpMean=False)

        # Quantiles
        # MAC.IP: Stats on src MAC-IP relationships
        # MI_quanstat = np.zeros((3 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     MI_quanstat[(i * 3):((i + 1) * 3)] = self.HT_MI_QUANT.update_get_1D_Stats(srcMAC + srcIP, timestamp,
        #                                                                    datagramSize,
        #                                                                    self.Lambdas[i], quantiles=[25, 50, 75])
        # if len(MI_quanstat) != len(self.Lambdas) * 3:
        #     print('issue in MI_quanstat')
        # # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        # HH_quanstat = np.zeros((3 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     HH_quanstat[(i * 3):((i + 1) * 3)] = self.HT_H_QUANT.update_get_1D2D_Stats(srcIP, dstIP, timestamp,
        #                                                                     datagramSize,
        #                                                                     self.Lambdas[i], quantiles=[25, 50, 75])
        # if len(HH_quanstat) != len(self.Lambdas) * 3:
        #     print('issue in HH_quanstat')
        #
        # # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        # HpHp_quanstat = np.zeros((3 * len(self.Lambdas, )))
        # if srcProtocol == 'arp':
        #     for i in range(len(self.Lambdas)):
        #         HpHp_quanstat[(i * 3):((i + 1) * 3)] = self.HT_Hp_QUANT.update_get_1D2D_Stats(srcMAC, dstMAC, timestamp,
        #                                                                            datagramSize, self.Lambdas[i],
        #                                                                            quantiles=[25, 50, 75])
        # else:  # some other protocol (e.g. TCP/UDP)
        #     for i in range(len(self.Lambdas)):
        #         HpHp_quanstat[(i * 3):((i + 1) * 3)] = self.HT_Hp_QUANT.update_get_1D2D_Stats(srcIP + srcProtocol,
        #                                                                            dstIP + dstProtocol, timestamp,
        #                                                                            datagramSize, self.Lambdas[i],
        #                                                                            quantiles=[25, 50, 75])
        # if len(HpHp_quanstat) != len(self.Lambdas) * 3:
        #     print('issue in HpHp_quanstat')
        # # DST stats
        # DT_MI_quanstat = np.zeros((3 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     DT_MI_quanstat[(i * 3):((i + 1) * 3)] = self.DT_MI_QUANT.update_get_1D_Stats(dstIP, timestamp, datagramSize,
        #                                                                       self.Lambdas[i], quantiles=[25, 50, 75])
        # if len(DT_MI_quanstat) != len(self.Lambdas) * 3:
        #     print('issue in DT_MI_quanstat')
        #return np.concatenate((MIstat, HHstat, HHstat_jit, HpHpstat, MI_flagstat_count, H_flagstat_count, HT_Hp_flagstat_count, MI_flagstat_mean, H_flagstat_mean, HT_Hp_flagstat_mean))  # concatenation of stats into one stat vector
        return HpHpstat
        return np.concatenate((MIstat, HHstat, HHstat_jit, HpHpstat, HtMiJitstat, HtHpJitstat, DT_MIstat, DtMiJitstat, MI_flagstat_mean, H_flagstat_mean, HT_Hp_flagstat_mean, DT_MI_flagstat_mean, MI_quanstat, HH_quanstat, HpHp_quanstat, DT_MI_quanstat))

    def updateGetStatsSecondHalf(self, IPtype, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, datagramSize, timestamp, tcpFlags=False):
        # Host BW: Stats on the srcIP's general Sender Statistics
        # Hstat = np.zeros((3*len(self.Lambdas,)))
        # for i in range(len(self.Lambdas)):
        #     Hstat[(i*3):((i+1)*3)] = self.HT_H.update_get_1D_Stats(srcIP, timestamp, datagramSize, self.Lambdas[i])

        # # MAC.IP: Stats on src MAC-IP relationships
        onedimensionalfeaturecount = 3
        twodimensionalfeaturecount = 7
        median=False
        # MIstat = np.zeros((3 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     MIstat[(i * 3):((i + 1) * 3)] = self.HT_MI.update_get_1D_Stats(srcMAC + srcIP, timestamp, datagramSize,
        #                                                                    self.Lambdas[i],median=median)
        # if len(MIstat) != len(self.Lambdas) * 3:
        #     print(self.Lambdas*3)
        #     print('issue in MIstat')
        #
        # # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        # HHstat = np.zeros((7 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     HHstat[(i * 7):((i + 1) * 7)] = self.HT_H.update_get_1D2D_Stats(srcIP, dstIP, timestamp, datagramSize,
        #                                                                     self.Lambdas[i],median=median)
        # if len(HHstat) != len(self.Lambdas) * 7:
        #     print('issue in HHstat')
        # # Host-Host Jitter:
        # HHstat_jit = np.zeros((4 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     HHstat_jit[(i * 4):((i + 1) * 4)] = self.HT_jit.update_get_1D_Stats(srcIP + dstIP, timestamp, 0,
        #                                                                         self.Lambdas[i], isTypeDiff=True,quantiles=[50])
        # if len(HHstat_jit) != len(self.Lambdas) * 4:
        #     print('issue in HHstat_jit')
        # # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        # HpHpstat = np.zeros((7 * len(self.Lambdas, )))
        # if srcProtocol == 'arp':
        #     for i in range(len(self.Lambdas)):
        #         HpHpstat[(i * 7):((i + 1) * 7)] = self.HT_Hp.update_get_1D2D_Stats(srcMAC, dstMAC, timestamp,
        #                                                                            datagramSize, self.Lambdas[i],median=median)
        # else:  # some other protocol (e.g. TCP/UDP)
        #     for i in range(len(self.Lambdas)):
        #         HpHpstat[(i * 7):((i + 1) * 7)] = self.HT_Hp.update_get_1D2D_Stats(srcIP + srcProtocol,
        #                                                                            dstIP + dstProtocol, timestamp,
        #                                                                            datagramSize, self.Lambdas[i],median=median)
        # if len(HpHpstat) != len(self.Lambdas) * 7:
        #     print('issue in HpHpstat')
        # # New code starts here
        # HtMiJitstat = np.zeros((4 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     HtMiJitstat[(i * 4):((i + 1) * 4)] = self.HT_MI_jit.update_get_1D_Stats(srcIP, timestamp, datagramSize,
        #                                                                    self.Lambdas[i], isTypeDiff=True,quantiles=[50])
        # if len(HtMiJitstat) != len(self.Lambdas) * 4:
        #     print('issue in HtMiJitstat')
        # HtHpJitstat = np.zeros((4* len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     HtHpJitstat[(i * 4):((i + 1) * 4)] = self.HT_Hp_jit.update_get_1D_Stats(srcIP+srcProtocol+dstIP+dstProtocol, timestamp, datagramSize,
        #                                                                    self.Lambdas[i], isTypeDiff=True,quantiles=[50])
        # if len(HtHpJitstat) != len(self.Lambdas) * 4:
        #     print('issue in HtHpJitstat')
        # # DST stats
        # DT_MIstat = np.zeros((onedimensionalfeaturecount * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     DT_MIstat[(i * onedimensionalfeaturecount):((i + 1) * onedimensionalfeaturecount)] = self.DT_MI.update_get_1D_Stats(dstIP, timestamp, datagramSize,
        #                                                                    self.Lambdas[i],median=median)
        # if len(DT_MIstat) != len(self.Lambdas) * 3:
        #     print('issue in DT_MIstat')
        # # DST Jitter
        # DtMiJitstat = np.zeros((4 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     DtMiJitstat[(i * 4):((i + 1) * 4)] = self.DT_MI_jit.update_get_1D_Stats(dstIP, timestamp, datagramSize,
        #                                                                             self.Lambdas[i], isTypeDiff=True,quantiles=[50])
        # if len(DtMiJitstat) != len(self.Lambdas) * 4:
        #     print('issue in DtMiJitstat')
        # Flag means
        MI_flagstat_mean = np.zeros((9 * len(self.Lambdas, )))
        if tcpFlags and tcpFlags != "":
            for i in range(len(self.Lambdas)):
                MI_flagstat_mean[(i * 9):((i + 1) * 9)] = self.HT_MI_FLAG_MEAN.update_get_1D_Stats(srcIP, timestamp, datagramSize, self.Lambdas[i], tcpFlags=tcpFlags, tcpMean=True)

        if len(MI_flagstat_mean) != len(self.Lambdas) * 9:
            print('issue in MI_flagstat_mean')
        H_flagstat_mean = np.zeros((9 * len(self.Lambdas, )))
        if tcpFlags and tcpFlags != "":
            for i in range(len(self.Lambdas)):
                H_flagstat_mean[(i * 9):((i + 1) * 9)] = self.HT_H_FLAG_MEAN.update_get_1D_Stats(srcIP+dstIP, timestamp, datagramSize,
                                                                                         self.Lambdas[i], tcpFlags=tcpFlags, tcpMean=True)
        if len(H_flagstat_mean) != len(self.Lambdas) * 9:
            print('issue in H_flagstat_mean')
        HT_Hp_flagstat_mean = np.zeros((9 * len(self.Lambdas, )))
        if tcpFlags and tcpFlags != "":
            for i in range(len(self.Lambdas)):
                HT_Hp_flagstat_mean[(i * 9):((i + 1) * 9)] = self.HT_Hp_FLAG_MEAN.update_get_1D_Stats(srcIP+srcProtocol+dstIP+dstProtocol, timestamp, datagramSize,
                                                                                         self.Lambdas[i], tcpFlags=tcpFlags, tcpMean=True)
        if len(HT_Hp_flagstat_mean) != len(self.Lambdas) * 9:
            print('issue in HT_Hp_flagstat_mean')
        DT_MI_flagstat_mean = np.zeros((9 * len(self.Lambdas, )))
        if tcpFlags and tcpFlags != "":
            for i in range(len(self.Lambdas)):
                DT_MI_flagstat_mean[(i * 9):((i + 1) * 9)] = self.DT_MI_FLAG_MEAN.update_get_1D_Stats(dstIP, timestamp, datagramSize,
                                                                                            self.Lambdas[i],
                                                                                            tcpFlags=tcpFlags, tcpMean=True)
        if len(DT_MI_flagstat_mean) != len(self.Lambdas) * 9:
            print('issue in DT_MI_flagstat_mean')
        # # Flag counts
        # MI_flagstat_count = np.zeros((8 * len(self.Lambdas, )))
        # if tcpFlags and tcpFlags != "":
        #     for i in range(len(self.Lambdas)):
        #         MI_flagstat_count[(i * 8):((i + 1) * 8)] = self.HT_MI_FLAG_COUNT.update_get_1D_Stats(srcIP, timestamp,
        #                                                                                            datagramSize,
        #                                                                                            self.Lambdas[i],
        #                                                                                            tcpFlags=tcpFlags,
        #                                                                                            tcpMean=False)
        # H_flagstat_count = np.zeros((8 * len(self.Lambdas, )))
        # if tcpFlags and tcpFlags != "":
        #     for i in range(len(self.Lambdas)):
        #         H_flagstat_count[(i * 8):((i + 1) * 8)] = self.HT_H_FLAG_COUNT.update_get_1D_Stats(srcIP + dstIP,
        #                                                                                          timestamp,
        #                                                                                          datagramSize,
        #                                                                                          self.Lambdas[i],
        #                                                                                          tcpFlags=tcpFlags,
        #                                                                                          tcpMean=False)
        # HT_Hp_flagstat_count = np.zeros((8 * len(self.Lambdas, )))
        # if tcpFlags and tcpFlags != "":
        #     for i in range(len(self.Lambdas)):
        #         HT_Hp_flagstat_count[(i * 8):((i + 1) * 8)] = self.HT_Hp_FLAG_COUNT.update_get_1D_Stats(
        #             srcIP + srcProtocol + dstIP + dstProtocol, timestamp, datagramSize,
        #             self.Lambdas[i], tcpFlags=tcpFlags, tcpMean=False)
        # DT_MI_flagstat_count = np.zeros((8 * len(self.Lambdas, )))
        # if tcpFlags and tcpFlags != "":
        #     for i in range(len(self.Lambdas)):
        #         DT_MI_flagstat_count[(i * 8):((i + 1) * 8)] = self.DT_MI_FLAG_COUNT.update_get_1D_Stats(dstIP, timestamp,
        #                                                                                               datagramSize,
        #                                                                                               self.Lambdas[i],
        #                                                                                               tcpFlags=tcpFlags,
        #                                                                                               tcpMean=False)

        # # Quantiles
        # # MAC.IP: Stats on src MAC-IP relationships
        # MI_quanstat = np.zeros((3 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     MI_quanstat[(i * 3):((i + 1) * 3)] = self.HT_MI_QUANT.update_get_1D_Stats(srcMAC + srcIP, timestamp,
        #                                                                    datagramSize,
        #                                                                    self.Lambdas[i], quantiles=[25, 50, 75])
        # if len(MI_quanstat) != len(self.Lambdas) * 3:
        #     print('issue in MI_quanstat')
        # # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        # HH_quanstat = np.zeros((3 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     HH_quanstat[(i * 3):((i + 1) * 3)] = self.HT_H_QUANT.update_get_1D2D_Stats(srcIP, dstIP, timestamp,
        #                                                                     datagramSize,
        #                                                                     self.Lambdas[i], quantiles=[25, 50, 75])
        # if len(HH_quanstat) != len(self.Lambdas) * 3:
        #     print('issue in HH_quanstat')
        #
        # # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        # HpHp_quanstat = np.zeros((3 * len(self.Lambdas, )))
        # if srcProtocol == 'arp':
        #     for i in range(len(self.Lambdas)):
        #         HpHp_quanstat[(i * 3):((i + 1) * 3)] = self.HT_Hp_QUANT.update_get_1D2D_Stats(srcMAC, dstMAC, timestamp,
        #                                                                            datagramSize, self.Lambdas[i],
        #                                                                            quantiles=[25, 50, 75])
        # else:  # some other protocol (e.g. TCP/UDP)
        #     for i in range(len(self.Lambdas)):
        #         HpHp_quanstat[(i * 3):((i + 1) * 3)] = self.HT_Hp_QUANT.update_get_1D2D_Stats(srcIP + srcProtocol,
        #                                                                            dstIP + dstProtocol, timestamp,
        #                                                                            datagramSize, self.Lambdas[i],
        #                                                                            quantiles=[25, 50, 75])
        # if len(HpHp_quanstat) != len(self.Lambdas) * 3:
        #     print('issue in HpHp_quanstat')
        # # DST stats
        # DT_MI_quanstat = np.zeros((3 * len(self.Lambdas, )))
        # for i in range(len(self.Lambdas)):
        #     DT_MI_quanstat[(i * 3):((i + 1) * 3)] = self.DT_MI_QUANT.update_get_1D_Stats(dstIP, timestamp, datagramSize,
        #                                                                       self.Lambdas[i], quantiles=[25, 50, 75])
        # if len(DT_MI_quanstat) != len(self.Lambdas) * 3:
        #     print('issue in DT_MI_quanstat')
        #return np.concatenate((MIstat, HHstat, HHstat_jit, HpHpstat, MI_flagstat_count, H_flagstat_count, HT_Hp_flagstat_count, MI_flagstat_mean, H_flagstat_mean, HT_Hp_flagstat_mean))  # concatenation of stats into one stat vector
        return np.concatenate((MI_flagstat_mean, H_flagstat_mean, HT_Hp_flagstat_mean, DT_MI_flagstat_mean))
        return np.concatenate((MIstat, HHstat, HHstat_jit, HpHpstat, HtMiJitstat, HtHpJitstat, DT_MIstat, DtMiJitstat, MI_flagstat_mean, H_flagstat_mean, HT_Hp_flagstat_mean, DT_MI_flagstat_mean, MI_quanstat, HH_quanstat, HpHp_quanstat, DT_MI_quanstat))

    def getNetStatHeaders(self):
        MIstat_headers = []
        Hstat_headers = []
        HHstat_headers = []
        HHjitstat_headers = []
        HpHpstat_headers = []

        for i in range(len(self.Lambdas)):
            MIstat_headers += ["MI_dir_"+h for h in self.HT_MI.getHeaders_1D(Lambda=self.Lambdas[i],ID=None)]
            HHstat_headers += ["HH_"+h for h in self.HT_H.getHeaders_1D2D(Lambda=self.Lambdas[i],IDs=None,ver=2)]
            HHjitstat_headers += ["HH_jit_"+h for h in self.HT_jit.getHeaders_1D(Lambda=self.Lambdas[i],ID=None)]
            HpHpstat_headers += ["HpHp_" + h for h in self.HT_Hp.getHeaders_1D2D(Lambda=self.Lambdas[i], IDs=None, ver=2)]
        return MIstat_headers + Hstat_headers + HHstat_headers + HHjitstat_headers + HpHpstat_headers
