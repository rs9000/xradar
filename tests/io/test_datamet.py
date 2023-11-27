from xradar.io.backends import datamet


def test_open_datamet():
    filename = r"C:\RADAR\RAW\2023\11\22\0300\VOL\H\LAURO"
    dtree = datamet.open_datamet_datatree(filename)
    assert dtree.origin == "LAURO"
