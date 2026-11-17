import os
import numpy as np
import pandas as pd
from PIL import Image
import re
import glob
import cv2
import json
from pathlib import Path

######### Konfiguration #########

LAB_ROOT = Path(r"C:\Users\messuser\Desktop\ProKINext")
HERE = Path(__file__).resolve().parent

if LAB_ROOT.exists():
    PATH_ROOT = LAB_ROOT
    PATH_DB = PATH_ROOT / "04_Database"
elif (HERE / "04_Database" / "Prozessdaten").exists():
    PATH_ROOT = HERE
    PATH_DB = PATH_ROOT / "04_Database"
elif (HERE.parent / "04_Database" / "Prozessdaten").exists():
    PATH_ROOT = HERE.parent
    PATH_DB = PATH_ROOT / "04_Database"
elif (HERE / "Prozessdaten").exists():
    PATH_ROOT = HERE
    PATH_DB = PATH_ROOT
elif (HERE.parent / "Prozessdaten").exists():
    PATH_ROOT = HERE.parent
    PATH_DB = PATH_ROOT
else:
    raise FileNotFoundError(f"No se encontró 'Prozessdaten' cerca de {HERE}")

PATH_force_signals: Path = PATH_DB / "Prozessdaten"
PATH_CONFIG_JSON: Path = PATH_DB / "Metadaten" / "parts.json"
SENSOR = ["Time", "Force", "Position", "Velocity", "Torque", "NH", "p_soll", "p_ist"]
PATH_QUALITY = PATH_DB / "Qualitaetsdaten" / "Qualitaetsdaten.txt"
PATH_IMAGES = PATH_DB / "Qualitaetsdaten" / "Bilder"

import models


def extract_id_of_file_name(file_name: str) -> int:
    """
    Extract id out of filename
    -----------
    Return id:int
    """
    match = re.search(r"ID(\d+)", file_name)
    if match:
        part_id = int(match.group(1))
        return part_id

    return None


def get_last_id(path: str = PATH_force_signals) -> int:
    """
    letzte eingetrage id ausgeben
    """
    ids = get_ids(path)
    return max(ids)


def get_ids(path: str | Path = PATH_force_signals) -> list[int]:
    """
    alle dateinamen auflisten
    """
    files = os.listdir(path)
    ids = [extract_id_of_file_name(file) for file in files]
    ids = [ID for ID in ids if ID is not None]
    return ids


def read_press_data(path: str | Path, ID: int) -> pd.DataFrame:
    file_name = os.path.join(path, f"ID{ID}_Pressendaten.txt")
    if os.path.exists(file_name):
        press_data = pd.read_csv(file_name, header=None, sep=";")
        press_data.columns = [
            "Time",
            "Force",
            "Position",
            "Velocity",
            "Torque",
            "NH",
            "p_soll",
            "p_ist",
        ]
        try:
            press_data.Time = pd.to_datetime(
                press_data.Time, format="%y.%m.%d %H:%M:%S,%f"
            )
        except ValueError:
            press_data.Time = pd.to_datetime(press_data.Time, format="%H:%M:%S,%f")

        return press_data
    else:
        print(f"Not found {file_name}")
        return None


def komma_2_punkt(file_path: str):
    """
    Ersetzt , druch . als Dezimaltrennzeichen
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as datei:
            text_inhalt = datei.read()

        text_inhalt_ersetzt = text_inhalt.replace(",", ".")

        with open(file_path, "w") as datei:
            datei.write(text_inhalt_ersetzt)
    else:
        print(f"FileError {file_path}")


def search_newest_id(path: str) -> int:
    """
    Neueste ID in Ordner finden
    """
    dateien = glob.glob(os.path.join(path, "*"))
    dateien.sort(key=os.path.getctime, reverse=True)
    filename = os.path.basename(dateien[0])
    part_id = extract_id_of_file_name(filename)
    return part_id


def generate_force_dataset(ids: list) -> np.ndarray:
    """
    Return: (samples,timesteps)
    """
    dataset = None
    for ID in ids:
        data = (
            read_press_data(PATH_force_signals, ID)["Force"].to_numpy().reshape(1, -1)
        )
        if dataset is None:
            dataset = data
        else:
            dataset = np.vstack((dataset, data))

    return dataset


def get_Quality_DataFrame() -> pd.DataFrame:
    return pd.read_csv(PATH_QUALITY, sep=";", decimal=".", dtype={0: int, 1: float})


def aktualisiere_winkel_csv(bauteil_id: int, winkel: float):
    df = get_Quality_DataFrame()
    maske = df.iloc[:, 0] == int(bauteil_id)

    if maske.any():
        df.loc[maske, df.columns[1]] = float(winkel)
    else:
        df.loc[len(df)] = [int(bauteil_id), float(winkel)]

    df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
    df.iloc[:, 0] = df.iloc[:, 0].astype(int)
    df.to_csv(PATH_QUALITY, sep=";", decimal=".", header=True, index=False)


class Bauteil:
    def __init__(self, ID: int):
        self.ID = ID

    @property
    def Prozessdaten(self):
        return read_press_data(PATH_force_signals, self.ID)

    @property
    def Qualitaetsdaten(self, Q: str = ""):
        df = get_Quality_DataFrame()
        maske = df.iloc[:, 0] == int(self.ID)

        if not maske.any():
            return None

        return float(df.loc[maske].iloc[0, 1])

    @Qualitaetsdaten.setter
    def Qualitaetsdaten(self, value: int):
        aktualisiere_winkel_csv(self.ID, value)

    @property
    def Machine(self) -> models.Machine:
        """
        OT,UT,Velocity,...
        """
        with open(PATH_CONFIG_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        machine_link = data[str(self.ID)]["machine"]
        return models.Machine.from_config(machine_link)

    @property
    def Material(self) -> models.Material:
        """
        OT,UT,Velocity,...
        """
        with open(PATH_CONFIG_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        machine_link = data[str(self.ID)]["material"]
        return models.Material.from_config(machine_link)

    @property
    def Werkzeug(self):
        """
        Werkzeugkonfiguration: Werkzeug Nr, KOmponenten, Eigenschaften, letzte Wartung?
        """
        pass

    @property
    def Time(self):
        return read_press_data(PATH_force_signals, self.ID)["Time"][0]

    @property
    def error(self):
        """Lädt Actions bei Bauteil ID aus Dokumentation"""
        with open(PATH_CONFIG_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data[str(self.ID)]["operator_error"]

    @property
    def actions(self):
        """Lädt Actions bei Bauteil ID aus Dokumentation"""
        with open(PATH_CONFIG_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data[str(self.ID)]["action"]

    @property
    def observations(self):
        """Lädt Actions bei Bauteil ID aus Dokumentation"""
        with open(PATH_CONFIG_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data[str(self.ID)]["observation"]