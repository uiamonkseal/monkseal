import csv
import fileinput
import os
import glob
import shutil
from os.path import exists
import pandas as pd
import scipy as scipy
from pydub import AudioSegment
from PIL import Image, ImageDraw
from PIL import ImageEnhance
import scipy.signal
from scipy.io import wavfile
from matplotlib import pyplot as plt



pixel_per_min = 7015
height = 244
class dataset():
    """
    def __init__(self, inputfile, wavPath):
        self.inputfile = inputfile
        self.data = pd.read_csv(inputfile, sep="\t", dtype=str)
        print(self.data)
        #self.data = inputfile
        self.wavPath = wavPath
        startMinNotEqualtoEndMin = 0
        for index, row in self.data.iterrows():
            try:
                start_min, start_sec = row["start_time"].split(":")
                start_min = int(start_min)
                start_sec = float(start_sec)

                end_min, end_sec = row["end_time"].split(":")
                end_min = int(end_min)
                end_sec = float(end_sec)
            except:
                print("Wronginput")
            if(start_min == end_min):
                #filepath,x1,y1,x2,y2,class_name
                print(row["file_name"]+"_"+str(start_min)+","+str(int(pixel_per_min/start_sec))+","+str(0)+","+str(int(pixel_per_min/end_sec))+","+str(height))
                #print(row["file_name"]+"_"+str(start_min)+","+str(pixel_per_min/start_sec)+","+str(0),+str(pixel_per_min/end_sec)+","+str(height))
            else:
                startMinNotEqualtoEndMin = startMinNotEqualtoEndMin+1
                self.data.drop(index)
            print("Vocals not included due to start minute not equal to end minute: " + str(startMinNotEqualtoEndMin))
    """

    def __init__(self,inputfile, wavPath):
        self.inputfile = inputfile
        self.data = pd.read_csv(inputfile, sep=",", dtype=str)
        print(self.data)
        #self.data = inputfile
        self.wavPath = wavPath
        startMinNotEqualtoEndMin = 0
        for index, row in self.data.iterrows():
            try:
                print(row)
                original_file, start, end, type = row["fname"].split("_")
                start_min = int(start[:2])
                end_min = int(end[:2])
                start_min = int(start_min)
                start_sec = float(int(start[2:4])+float(int(start[4:6]))/100)
                end_sec = float(int(end[2:4])+float(int(end[4:6]))/100)

            except:
                print("Wronginput")
            if(start_min == end_min):
                #print("1")
                i = 0
                #filepath,x1,y1,x2,y2,class_name
                #print(row["fname"]+"_"+str(start_min)+","+str(int(pixel_per_min/start_sec))+","+str(0)+","+str(int(pixel_per_min/end_sec))+","+str(height))
                #print(row["file_name"]+"_"+str(start_min)+","+str(pixel_per_min/start_sec)+","+str(0),+str(pixel_per_min/end_sec)+","+str(height))
            else:
                startMinNotEqualtoEndMin = startMinNotEqualtoEndMin+1
                self.data = self.data.drop(index)
        print("Vocals not included due to start minute not equal to end minute: " + str(startMinNotEqualtoEndMin))

    def ListOfOriginalFilesNotExists(self):
        list = []
        for index, row in self.data.iterrows():
            if not exists("D:\\monksounds\\originalfiles\\"+row["fname"].split("_")[0].split(".")[0]+"_"+row["fname"].split("_")[0].split(".")[1]+".wav"):
                list.append(row["fname"].split("_")[0]) if row["fname"].split("_")[0] not in list else list
        for i in list:
            print(i)


    def copyRelevantFiles(self,inputFolder,outputFolder):
        for index,row in self.data.iterrows():
            filename = row['fname'].split("_")[0].split(".")[0]+"_"+row['fname'].split("_")[0].split(".")[1]
            start_min = int(row['fname'].split("_")[1][:2])
            dropbox = "D:\\Dropbox\\Dropbox\\Monk Seal AI project\\Data\\Sound files\\HMS acoustic files\\SoundTrap Recordings August 2017-September 2018"
            if not exists(outputFolder+"//"+filename+"_"+str(start_min)+".wav"):
                try:
                    shutil.copy(inputFolder+"//"+filename+"_"+str(start_min)+".wav", outputFolder+"//"+filename+"_"+str(start_min)+".wav")
                except:
                    print("File not found")
                    if not exists("D://newfiles"+"//"+row['fname'].split("_")[0]+".sud"):
                        shutil.copy(dropbox+"//"+row['fname'].split("_")[0]+".sud", "D://newfiles"+"//"+row['fname'].split("_")[0].split(".")[0]+"_"+row['fname'].split("_")[0].split(".")[1]+".sud")


    def splitFiles(self,outputPath):
        import glob, os
        os.chdir(self.wavPath)
        for file in glob.glob("*.wav"):
            audio = AudioSegment.from_file(file)
            i = int(audio.duration_seconds/60)
            j = 0
            while j < i:
                t1 = j*60000
                t2 = (j*60000)+60000
                split_filename = file.split(".")[0]+"_"+str(j)+"."+file.split(".")[1]
                j = j+1
                split_audio = audio[t1:t2]
                split_audio.export(outputPath+"//"+split_filename, format="wav")

    def createSpectrogramFromWav(self,data,fs):
        f, t, Sxx, = scipy.signal.spectrogram(data, fs=fs, window='hanning', nperseg=4096,
                                              noverlap=round(4096 * 0.9), mode='complex')
        f = f[:200]
        Sxx = Sxx[:265]  # Filter out top frequancy
        Sxx = Sxx[21:]  # Filter out noise 21px
        Sxx = Sxx[::-1]  # Flip array to make lower freq at bottom
        Im = Image.fromarray(abs(Sxx) * 255)

        Im = Im.convert("L")
        # Im = Im.convert("RGB")

        ime = ImageEnhance.Contrast(Im)
        Im = ime.enhance(1.8)
        # plt.imshow
        plt.imshow(Im)
        plt.show()
        #Im.show()
        return Im

    #Returns start and end pixels for a list of a numpy-frame
    def getPixelCoordinatesFromSingleFrame(self,dataframe):
        #start = int(float(dataframe.start_time.split(":")[1])*(pixel_per_min/60))
        #end = int(float(dataframe.end_time.split(":")[1])*(pixel_per_min/60))
        start = []

        start = int(float(float(dataframe['fname'].split("_")[1][2:4]) + ((float(dataframe['fname'].split("_")[1][4:6]) / 100)))*(pixel_per_min/60))
        end = int(float(float(dataframe['fname'].split("_")[2][2:4]) + ((float(dataframe['fname'].split("_")[2][4:6]) / 100)))*(pixel_per_min/60))
        #return dataframe.file_name.split(".")[0]+"_"+dataframe.file_name.split(".")[1], dataframe.start_time.split(":")[0], [start,0,end,height]
        return dataframe['fname'].split('_')[0].split('.')[0]+"_"+dataframe['fname'].split('_')[0].split('.')[1], dataframe['fname'].split('_')[1][:2], [start,0,end,height]

def line_prepender(filename):
    with open(filename, 'r+') as f:
        for count, line in enumerate(f):
            pass
        content = f.read()
        f.seek(0, 0)
#        print(content)
        f.write(str(count) + '\n' + content)

def line_pre_adder(filename, line_to_prepend):
    f = fileinput.input(filename, inplace=1)
    for count, line in enumerate(f):
        pass
    for xline in f:
        if f.isfirstline():
            print(str(count).rstrip('\r\n') + '\n' + xline,)
        else:
            print(xline,)

def insert(originalfile):
    once = True
    with open(originalfile, 'r+') as fp:
        lines = fp.readlines()  # lines is list of line, each element '...\n'
        lines.insert(0, str(len(lines))+"\n")  # you can use any index if you know the line index
        fp.seek(0)  # file pointer locates at the beginning to write the whole file again
        fp.writelines(lines)  # write whole lists again to the same file
        fp.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #All needs to be run in sequence.
    #First split each long wav-file into 1 minute spectrograms.
    #Keep only relevant files (at least one vocalization per minute)
    #Create spectrograms for relevant files.
    #Create individual annotations-file for each spectrogram that consists of start and end posistion of vocalizations.


    #dataset = dataset("../labels.txt","D://monksounds//originalfiles")
    dataset = dataset("../labels.txt", "D://monksounds//splittedfiles//relevant")
    #dataset.splitFiles("D://monksounds//splittedfiles2")
    #dataset.copyRelevantFiles("D:\\monksounds\\splittedfiles","D:\\monksounds\\splittedfiles\\relevant2")
    #sample = dataset.data.sample(60)
    dataset.ListOfOriginalFilesNotExists()
    for index, row in dataset.data.iterrows():
        #samplingFrequency, signalData = wavfile.read("D:\\monksounds\\splittedfiles\\" + row["file_name"].split(".")[0] + "_" + row["file_name"].split(".")[1] + "_" + str(int(row["start_time"].split(":")[0])) + ".wav")
        #Im = dataset.createSpectrogramFromWav(signalData,samplingFrequency)
        #img1 = ImageDraw.Draw(Im)

        filename, minute, coordinates = dataset.getPixelCoordinatesFromSingleFrame(row)
        shape = coordinates
        if coordinates[0] != coordinates[2]:
            with open(filename+"_"+str(int(minute))+".txt", 'a+') as f:
                f.write(str(coordinates[0])+","+str(coordinates[1])+","+str(coordinates[2])+","+str(coordinates[3])+"\n")
                f.close()
        #shape = [(int(float(row["start_time"].split(":")[1])*(pixel_per_min/60)), 40), int(float(row["end_time"].split(":")[1])*(pixel_per_min/60)), h - 10]
        #print(shape)
        #print(row)
        #img1.rectangle(shape,outline="#ffff33")
        #Im.show()
        os.chdir(".")
    for file in glob.glob("*.txt"):
        insert(file)

