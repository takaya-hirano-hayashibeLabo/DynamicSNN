import tonic
from pathlib import Path


def main():

    #>> NMNISTデータ >>
    print("\n\033[92mNMNIST\033[0m"+"==================================")
    sensor_size = tonic.datasets.NMNIST.sensor_size
    original_datapath=str(Path(__file__).parent)
    trainset = tonic.datasets.NMNIST(save_to=original_datapath, train=True)
    testset = tonic.datasets.NMNIST(save_to=original_datapath,  train=False)

    print("Sensor size: ",sensor_size)
    print("Trainset size:", len(trainset),"Testset size:", len(testset))

    print("Sequence Length: ",trainset[0][0].shape) #1つのイベントデータの時系列長さ    
    # testsetのクラス数を表示
    num_classes = len(set(testset.targets))
    print("Number of classes:", num_classes)

    print(f"Event data exp: {testset[0][0][0]}") # [px, py , timestamp, event(0 or 1)]
    #<< NMNISTデータ <<


    #>> DVSGesture >>
    print("\n\033[92mDVSGeuture\033[0m"+"==================================")
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    original_datapath=str(Path(__file__).parent)
    trainset = tonic.datasets.DVSGesture(save_to=original_datapath, train=True)
    testset = tonic.datasets.DVSGesture(save_to=original_datapath,  train=False)

    print("Sensor size: ",sensor_size)
    print("Trainset size:", len(trainset),"Testset size:", len(testset))
    print("Sequence Length: ",trainset[0][0].shape) #1つのイベントデータの時系列長さ    
    # testsetのクラス数を表示
    num_classes = len(set(testset.targets))
    print("Number of classes:", num_classes)

    print(f"Event data exp: {testset[0][0][0]}") #[px, py, event(True/False), timestamp]
    print(testset.data)
    #<< DVSGesture <<


if __name__=="__main__":
    main()