import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from TDSpy.sn_TDS import sn_TDS_no_feature_extraction as TDS
from TDSpy.feature_extraction.sn_getEEGBandPower import sn_getEEGBandPower
from TDSpy.feature_extraction.sn_getVariance import sn_getVariance
from TDSpy.tools.sn_plotTDS import plot_TDS
from TDSpy.tools.edf_reader import read_all_EDF_channels


# Подготовка: заранее скачайте файлы
#   https://physionet.org/content/ucddb/1.0.0/ucddb002.rec
#   https://physionet.org/content/ucddb/1.0.0/ucddb002_stage.txt


def main():
    sleep_stage = 0  # код стадии сна: 0 = бодрствование
    dur = 16_000  # анализируем первые 16 000 секунд записи

    # 1. Чтение EDF-файла
    data_dict, data_dict_sampling_rate = read_all_EDF_channels(
        "ucddb002.rec", startrecord=0, endrecord=dur
    )

    # 2. Чтение файла со стадиями сна
    stages = pd.read_csv("ucddb002_stage.txt", header=None).to_numpy()

    # 3. Индексы только интересующей стадии
    stage_idx = np.where(stages == sleep_stage)[0]
    stage_idx = stage_idx[stage_idx < dur / 30]  # каждая строка = 30 с

    # 4. Обработка ЭКГ → частота сердцебиения (1 Гц)
    ecg_proc = nk.ecg_process(
        data_dict["ECG"],
        sampling_rate=data_dict_sampling_rate["ECG"],
        method="neurokit",
    )
    hr_signal_resampled = nk.signal_resample(
        ecg_proc[0]["ECG_Rate"],
        sampling_rate=data_dict_sampling_rate["ECG"],
        desired_sampling_rate=1,
        method="interpolation",
    )

    # 5. Обработка дыхания (Flow) → частота дыхания (1 Гц)
    rsp_rate = nk.rsp_rate(data_dict["Flow"], sampling_rate=8, method="trough")
    rsp_signal_resampled = nk.signal_resample(
        rsp_rate,
        sampling_rate=data_dict_sampling_rate["Flow"],
        desired_sampling_rate=1,
        method="interpolation",
    )

    # 6. ЭМГ: дисперсия в окне 1 с
    emg_var = sn_getVariance(data_dict["EMG"], sf=data_dict_sampling_rate["EMG"])

    # 7. ЭОГ: дисперсия в окне 1 с
    eog_var = sn_getVariance(
        data_dict["Lefteye"], sf=data_dict_sampling_rate["Lefteye"]
    )

    # 8. ЭЭГ: мощности в пяти диапазонах
    fpb, _ = sn_getEEGBandPower(
        data_dict["C3A2"],
        sf=data_dict_sampling_rate["C3A2"],
        bandlimits=np.array([[0.5, 4, 8, 12, 16], [3.5, 7.5, 11.5, 15.5, 19.5]]),
    )

    # 9. Готовим словарь признаков
    features = {
        "HR": hr_signal_resampled,
        "Resp": rsp_signal_resampled,
        "Chin": emg_var,
        "Eye": eog_var,
        "Delta": fpb[:, 0],
        "Theta": fpb[:, 1],
        "Alpha": fpb[:, 2],
        "Sigma": fpb[:, 3],
        "Beta": fpb[:, 4],
    }

    # 10. Расчёт TDS
    tds, combination, _ = TDS(data_dict=features)

    # 11. Сигналы — обзорный график
    nk.signal_plot(pd.DataFrame(features), subplots=True)
    plt.savefig("ucd_channels.png", dpi=150)
    plt.show()

    # 12. Оставляем только окна требуемой стадии сна
    tds_stage = tds[:, stage_idx[:-1]]

    # 13. Матрица link-strength (тепловая карта)
    plot_TDS(tds_stage, combination)


if __name__ == "__main__":
    main()
