import re
import requests


# 任意の実験データの取得と前処理を行う関数
def clearning(file_input, file_output, start, end):
    print("-----Open File-----")
    with open(file_input) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("#S " + str(start)):
                i1 = i
            if line.startswith("#S " + str(end + 1)):
                i2 = i

        lines = [line.strip() for line in lines]
        result = [
            "Energy  MonoTheta  TwoTheta  Theta  Chi  Phi  Epoch  \
            Seconds  I1 PA0  PA1  Ch2  Ch3  XP1  TB  \
            TC  Att  RC  Timepix1  Timepix2  Timepix3\
            Timepix4  TP_max  Monitor  Detector"
        ]

        for line in lines[i1:i2]:
            result.append(re.sub(r"^\s*#.*$", "", line, flags=re.MULTILINE))

        result = [s for s in result if s != ""]
        result = map(lambda x: x + "\n", result)

        with open(file_output, mode="w") as f:
            f.writelines(result)
    print("-----Saved-----")


# データベースからのダウンロードを行う関数
def download(url, file_name):
    response = requests.get(url)

    if response.status_code == 200:
        with open(file_name, "wb") as f:
            for chunk in response.iter_content(100000):
                f.write(chunk)
        print("Completed")
    else:
        print("Failed")
