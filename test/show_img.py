import numpy as np
import rasterio
import matplotlib.pyplot as plt

def getArrFlood(fname):
    return rasterio.open(fname).read()


# Paraguay_921838_LabelHand.tif  USA_933610_LabelHand.tif  Paraguay_76868_LabelHand Sri-Lanka_101973_LabelHand.tif

# im_fname1 = '/home/amax/SSD1/zjzRoot/project/data/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Spain_7370579_S1Hand.tif'
# im_fname2 = '/home/amax/SSD1/zjzRoot/project/data/sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/Spain_7370579_S2Hand.tif'


def show_every_channel(np2, np3):
    rows = 4
    cols = 4

    # 创建一个大图，绘制15个子图
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))

    # 遍历每个通道，并绘制图像到对应的子图
    for i in range(15):
        ax = axes[i // cols, i % cols]  # 计算当前通道对应的子图
        ax.imshow(np3[i], cmap='gray')  # 显示为灰度图
        ax.axis('off')  # 去掉坐标轴
        ax.set_title(f'Channel {i+1}')  # 标题标明通道号


    # 堆叠为 RGB 图像 (NIR, Red, Green)
    fcc_image = np.stack([np2[3], np2[2], np2[1]], axis=-1)

    # 归一化到 [0, 1] 范围，方便显示
    fcc_image = (fcc_image - fcc_image.min()) / (fcc_image.max() - fcc_image.min())

    ax = axes[3,3]  # 计算当前通道对应的子图
    ax.imshow(fcc_image)  # 显示rgb
    ax.axis('off')  # 去掉坐标轴
    ax.set_title(f'Channel rgb')  # 标题标明通道号

    # 调整子图间距
    plt.tight_layout()
    plt.show()

    pass


def normalize(array):
    """将数组的值归一化到 0-1 之间"""
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)


def show_rbg(np2, save_name):
    # 创建一个大图，绘制15个子图
    fig, ax = plt.subplots(figsize=(16, 16))

    # 将波段数据类型转换为 float32
    swir = np2[12]
    nir = np2[7]
    green = np2[3]
    blue = np2[2]

    # 归一化
    swir_norm = normalize(swir)
    nir_norm = normalize(nir)
    green_norm = normalize(green)
    blue_norm = normalize(blue)

    # 堆叠为 RGB 图像
    fcc_image = np.stack([np2[3], np2[7], np2[2]], axis=-1)

    # 计算 Blue 通道: (NIR + Green + Blue) / 3
    blue_channel = (nir_norm + green_norm + blue_norm) / 3

    # 堆叠为 FCC 图像 (SWIR, NIR, (NIR+Green+Blue)/3)
    fcc_image = np.dstack((swir_norm, nir_norm, blue_channel))

    plt.imsave("data_visible/fcc_"+ save_name +".png", fcc_image)

    # ax.imshow(fcc_image)  # 显示rgb
    # ax.axis('off')  # 去掉坐标轴
    # ax.set_title(f'Channel rgb')  # 标题标明通道号
    #
    # # 调整子图间距
    # plt.tight_layout()
    # plt.show()

    pass


def show_real_rbg(np2, save_name):
    # 创建一个大图，绘制15个子图
    fig, ax = plt.subplots(figsize=(16, 16))

    # 将波段数据类型转换为 float32
    red = np2[1]
    green = np2[3]
    blue = np2[2]

    # 归一化
    # red_norm = normalize(red)
    # green_norm = normalize(green)
    # blue_norm = normalize(blue)

    rbg_image = np.dstack((red, green, blue))
    rbg_image  = normalize(rbg_image)

    plt.imsave("data_visible/rgb_"+ save_name +".png", rbg_image)

    # ax.imshow(rbg_image)  # 显示rgb
    # ax.axis('off')  # 去掉坐标轴
    # ax.set_title(f'Channel rgb')  # 标题标明通道号
    #
    # # 调整子图间距
    # plt.tight_layout()
    # plt.show()

    pass


def show_label(np_label, save_name):
    plt.imsave("data_visible/label_"+ save_name +".png", np_label[0], cmap='gray')


if __name__ == '__main__':

    data_list = [
        "Ghana_103272",
        "Ghana_24858",
        "Ghana_147015",
        "Ghana_953791",
        "Ghana_154838",
        "Ghana_134751",
        "Ghana_61925",
        "Ghana_156478",
        "Ghana_144050",
        "Ghana_49890",
        "Ghana_97516",
        "Ghana_168875",
        "Ghana_141910",
        "Ghana_146222",
        "Ghana_11745",
        "Ghana_161233",
        "Ghana_7496",
        "Ghana_128663",
        "Ghana_264787",
        "Ghana_597288",
        "Ghana_8090",
        "Ghana_187318",
        "Ghana_362274",
        "Ghana_1089161",
        "Ghana_887131",
        "Ghana_247288",
        "Ghana_194723",
        "Ghana_26376",
        "Ghana_45911",
        "Ghana_83200",
        "Ghana_135389",
        "India_285297",
        "India_1072277",
        "India_324254",
        "India_177319",
        "India_391908",
        "India_166709",
        "India_534470",
        "India_773682",
        "India_956930",
        "India_979278",
        "India_664410",
        "India_747970",
        "India_207862",
        "India_500266",
        "India_179238",
        "India_287642",
        "India_774689",
        "India_383430",
        "India_373039",
        "India_698338",
        "India_1017769",
        "India_981708",
        "India_91379",
        "India_59460",
        "India_624341",
        "India_566697",
        "India_804466",
        "India_103447",
        "India_136196",
        "India_842775",
        "India_25540",
        "India_56450",
        "India_135434",
        "India_652725",
        "India_943439",
        "India_73419",
        "India_591549",
        "India_273873",
        "India_707886",
        "India_265762",
        "Mekong_221789",
        "Mekong_52610",
        "Mekong_596495",
        "Mekong_1439641",
        "Mekong_1191208",
        "Mekong_774566",
        "Mekong_213094",
        "Mekong_119477",
        "Mekong_269835",
        "Mekong_846007",
        "Mekong_1111068",
        "Mekong_1282475",
        "Mekong_16233",
        "Mekong_342411",
        "Mekong_1395593",
        "Mekong_922373",
        "Mekong_1396181",
        "Mekong_1248200",
        "Nigeria_529525",
        "Nigeria_143329",
        "Nigeria_439488",
        "Nigeria_707067",
        "Nigeria_81933",
        "Nigeria_35845",
        "Nigeria_598959",
        "Nigeria_952958",
        "Nigeria_78061",
        "Nigeria_600295",
        "Pakistan_909806",
        "Pakistan_132143",
        "Pakistan_712873",
        "Pakistan_35915",
        "Pakistan_246510",
        "Pakistan_667363",
        "Pakistan_366265",
        "Pakistan_548910",
        "Pakistan_740461",
        "Pakistan_211386",
        "Pakistan_401863",
        "Pakistan_246718",
        "Pakistan_474121",
        "Pakistan_65724",
        "Pakistan_1036366",
        "Pakistan_760290",
        "Paraguay_721886",
        "Paraguay_62897",
        "Paraguay_36015",
        "Paraguay_12870",
        "Paraguay_126224",
        "Paraguay_215904",
        "Paraguay_54421",
        "Paraguay_247656",
        "Paraguay_179624",
        "Paraguay_1029042",
        "Paraguay_246154",
        "Paraguay_892933",
        "Paraguay_11869",
        "Paraguay_921838",
        "Paraguay_24341",
        "Paraguay_605682",
        "Paraguay_997480",
        "Paraguay_148318",
        "Paraguay_191503",
        "Paraguay_1076204",
        "Paraguay_198534",
        "Paraguay_470303",
        "Paraguay_822142",
        "Paraguay_795075",
        "Paraguay_339807",
        "Paraguay_1056717",
        "Paraguay_403081",
        "Paraguay_149787",
        "Paraguay_36146",
        "Paraguay_792268",
        "Paraguay_482517",
        "Paraguay_212687",
        "Paraguay_791364",
        "Paraguay_224845",
        "Paraguay_149830",
        "Paraguay_48673",
        "Paraguay_44682",
        "Paraguay_989230",
        "Paraguay_225187",
        "Somalia_989553",
        "Somalia_93023",
        "Somalia_371421",
        "Somalia_230192",
        "Somalia_886726",
        "Somalia_992457",
        "Somalia_7931",
        "Somalia_195014",
        "Somalia_32375",
        "Somalia_1087508",
        "Somalia_295782",
        "Somalia_970508",
        "Somalia_626316",
        "Somalia_1068756",
        "Somalia_205466",
        "Spain_5923267",
        "Spain_7786924",
        "Spain_7856615",
        "Spain_4915752",
        "Spain_2472849",
        "Spain_8104659",
        "Spain_5816638",
        "Spain_2523247",
        "Spain_6199994",
        "Spain_8154154",
        "Spain_1167260",
        "Spain_2938657",
        "Spain_5678382",
        "Spain_716716",
        "Spain_3285448",
        "Spain_8199661",
        "Spain_337094",
        "Spain_2594119",
        "Sri-Lanka_152185",
        "Sri-Lanka_55568",
        "Sri-Lanka_92824",
        "Sri-Lanka_847275",
        "Sri-Lanka_845821",
        "Sri-Lanka_14484",
        "Sri-Lanka_1038087",
        "Sri-Lanka_178753",
        "Sri-Lanka_163406",
        "Sri-Lanka_233609",
        "Sri-Lanka_135713",
        "Sri-Lanka_748447",
        "Sri-Lanka_579082",
        "Sri-Lanka_916628",
        "Sri-Lanka_883641",
        "Sri-Lanka_52223",
        "Sri-Lanka_400518",
        "Sri-Lanka_523539",
        "Sri-Lanka_49764",
        "Sri-Lanka_653336",
        "Sri-Lanka_956740",
        "Sri-Lanka_249079",
        "Sri-Lanka_120804",
        "Sri-Lanka_551926",
        "USA_994009",
        "USA_66026",
        "USA_11422",
        "USA_955053",
        "USA_114964",
        "USA_693819",
        "USA_86502",
        "USA_438959",
        "USA_181263",
        "USA_605492",
        "USA_115033",
        "USA_375183",
        "USA_347609",
        "USA_908474",
        "USA_211406",
        "USA_831672",
        "USA_664261",
        "USA_770353",
        "USA_217598",
        "USA_638521",
        "USA_387945",
        "USA_224165",
        "USA_231124",
        "USA_170264",
        "USA_58086",
        "USA_1010394",
        "USA_198411",
        "USA_486103",
        "USA_260929",
        "USA_179917",
        "USA_84195",
        "USA_604222",
        "USA_354981",
        "USA_232060",
        "USA_655230",
        "USA_652955",
        "USA_1068362",
        "USA_806273",
        "USA_788696",
        "USA_348639",
        "USA_181503",
    ]  # 训练集合

    data_list = [
        "Mekong_1149855",
        "Mekong_977338",
        "Mekong_474783",
        "Mekong_293769",
        "Mekong_1413877",
        "Mekong_98310",
        "Nigeria_31096",
        "Nigeria_984831",
        "Nigeria_1095404",
        "Nigeria_820924",
        "Pakistan_43105",
        "Pakistan_94095",
        "Pakistan_210595",
        "Pakistan_1027214",
        "Pakistan_336228",
        "Pakistan_9684",
        "Paraguay_305760",
        "Paraguay_648632",
        "Paraguay_172476",
        "Paraguay_581976",
        "Paraguay_284928",
        "Paraguay_1019808",
        "Paraguay_76868",
        "Paraguay_252217",
        "Paraguay_205585",
        "Paraguay_7894",
        "Paraguay_896458",
        "Paraguay_657443",
        "Paraguay_934240",
        "Paraguay_153941",
        "Somalia_12849",
        "Somalia_256539",
        "Somalia_61368",
        "Somalia_649376",
        "Somalia_167787",
        "Spain_1199913",
        "Spain_8372658",
        "Spain_7604243",
        "Spain_6537196",
        "Spain_4282030",
        "Spain_8565131",
        "Sri-Lanka_85652",
        "Sri-Lanka_63307",
        "Sri-Lanka_612594",
        "Sri-Lanka_132922",
        "Sri-Lanka_236030",
        "Sri-Lanka_31559",
        "Sri-Lanka_236628",
        "Sri-Lanka_101973",
        "Sri-Lanka_321316",
        "USA_826217",
        "USA_741073",
        "USA_275372",
        "USA_19225",
        "USA_366607",
        "USA_308150",
        "USA_1039203",
        "USA_251323",
        "USA_1082482",
        "USA_225017",
        "USA_986268",
        "USA_646878",
        "USA_761032",
        "USA_741178",
    ]
    # data_list = ['Mekong_1439641', 'Spain_2938657', 'Spain_7370579']

    for item in data_list:

        im_fname1 = '/home/amax/SSD1/zjzRoot/project/data/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/' + item + '_S1Hand.tif'
        im_fname2 = '/home/amax/SSD1/zjzRoot/project/data/sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/' + item + '_S2Hand.tif'
        im_flabel = '/home/amax/SSD1/zjzRoot/project/data/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/' + item + '_LabelHand.tif'

        np1 = np.nan_to_num(getArrFlood(im_fname1))
        np2 = np.nan_to_num(getArrFlood(im_fname2))
        np3 = np.concatenate((np1, np2), axis=0)
        np_label = np.nan_to_num(getArrFlood(im_flabel))

        show_rbg(np2, item)
        show_real_rbg(np2, item)
        show_label(np_label, item)