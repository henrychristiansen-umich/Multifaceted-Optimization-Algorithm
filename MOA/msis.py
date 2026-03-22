import pymsis
import numpy as np
from datetime import datetime, timedelta

if __name__ == "__main__":
    year_doy = 2000001
    sod = 43168.0 
    year = year_doy // 1000
    doy = year_doy % 1000
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1, seconds=sod)

    out = pymsis.calculate(
        np.datetime64(dt),
        79.671592712,
        -0.001446289127,
        443.863708,
        131.935363769531,
        166.199996948242,
        [[30, 18, 27, 39, 56, 35.875, 11.125]],
        version=0,
        geomagnetic_activity=-1,
        )
    out = np.squeeze(out)
    print(out[pymsis.Variable.MASS_DENSITY] * 1e12,"e-03")

