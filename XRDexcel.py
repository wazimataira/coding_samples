import pandas as pd
from tkinter import filedialog
import os

#ファイルブラウズ
def filebrowse():
    wdtitle="読み込むデータファイルを選択"
    fltyp=[("XRD ras File",".ras"),("Text File",".txt"),("Dat File",".dat")]
    file_path=filedialog.askopenfilename(filetypes=fltyp,initialdir='./',title=wdtitle,multiple=True)
    print("load here {}".format(file_path))
    return file_path[0]

#保存ディレクトリの選択
def dir_brows():
    tit="保存フォルダを選択"
    inidir='./'
    dir_path=filedialog.askdirectory(initialdir = inidir,title=tit)
    print("save here {}".format(dir_path))
    return dir_path
    

class MakeExcelFile:
    def __init__(self):
        self.idx=[]
        self.headder=[]
        self.content=[]
        self.df=pd.DataFrame()
        self.loadpath=filebrowse()
        self.savedir=dir_brows()
    
    #データのロード
    def loaddata(self):
        #ファイルをopenしてリストに
        with open(self.loadpath,'r') as f:
            for i,line in enumerate(f):
                self.idx.append(i)
                if 245<i<5247 :
                    self.content.append([float(ln) for ln in line.split()])
                elif i>5247:
                    pass
                else:
                    self.headder.append(line)

    #Pandasデータフレーム型にデータを格納
    def shaping(self):
        df_head=pd.DataFrame(self.headder,columns=['A'])
        df_content=pd.DataFrame(data=self.content,columns=['A','B','C'],dtype=float)
        self.df=df_head.append(df_content,ignore_index=True)

    #Excelファイルの出力
    def save(self):
        file_name=os.path.splitext(os.path.basename(self.loadpath))[0]
        excel_name=self.savedir+"/"+file_name+'.xlsx'
        print("make file here {}".format(excel_name))
        self.df.to_excel(excel_name,index=False,header=False)

def main():
    print("-----make excel file-----")
    excel=MakeExcelFile()
    print("-----load-----")
    excel.loaddata()
    excel.shaping()
    print("-----save-----")
    excel.save()
    print("-----finished-----")

if __name__ =='__main__':
    main()
    
