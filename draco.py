import math
import time
import sys
import re
import numpy as np
from pathlib import Path
import os.path
import vtk


def get_tmpdir():
    if Path('/scratch').exists():
        return '/scratch'
    if Path('/var/tmp').exists():
        return '/var/tmp'
    return './'


argv = sys.argv
if len(argv) < 6:
    print("python draco.py x.f32.dat y.f32.dat z.f32.dat 7852 1037 -qp 10 -ts 0")
    print("-qp 10 is optional, means use 10 bits for quantization" )
    print("-ts 0 is optional, means compress frame 0 instead of all the frames" )
    exit()

filenamex = argv[1]
filenamey = argv[2]
filenamez = argv[3]
timestep = int(argv[4])
r1 = int(argv[5])
qps = list(range(1, 31))
tss = list(range(timestep))
argc = 6
while argc < len(argv):
    if argv[argc].startswith('-qp'):
        qps = [int(argv[argc + 1])]
        argc += 2
    elif argv[argc].startswith('-ts'):
        tss = [int(argv[argc + 1])]
        argc += 2

print(qps)
print('single frame' if len(tss) == 1 else 'multiple frame')

x = np.fromfile(Path(filenamex), dtype=np.float32).reshape(timestep, r1)
y = np.fromfile(Path(filenamey), dtype=np.float32).reshape(timestep, r1)
z = np.fromfile(Path(filenamez), dtype=np.float32).reshape(timestep, r1)

HOME = str(Path.home())
draco_exe_folder = HOME + '/code/draco/build/'

for qp in qps:
    total_compressed_size = 0
    total_compress_time = 0
    total_decompress_time = 0
    max_diffx = 0
    max_diffy = 0
    max_diffz = 0
    errorx = 0
    errory = 0
    errorz = 0

    for ts in tss:
        src_file = "{}/draco/{}-{}.ply".format(get_tmpdir(), Path(filenamex).absolute().parent.name, ts)
        compressed_file = src_file + '.draco'
        decompressed_file = compressed_file + '.draco.ply'
        if not Path(src_file).exists():
            Path(src_file).parent.mkdir(exist_ok=True, parents=True)
            points = vtk.vtkPoints()
            for pt in range(r1):
                points.InsertNextPoint(x[ts][pt], y[ts][pt], z[ts][pt])

            data_save = vtk.vtkPolyData()
            data_save.SetPoints(points)

            writer = vtk.vtkPLYWriter()
            writer.SetInputData(data_save)
            writer.SetFileName(src_file)
            writer.Write()

        exe = "DRACO_PSNR=TRUE {}/draco_encoder -point_cloud -i {} -o {} -qp {} 2>&1".format(
            draco_exe_folder, src_file, compressed_file, qp)
        # print(exe)
        out = os.popen(exe).read()
        # print('out=', out)
        # compress_time = float(re.findall('compression time = ([.0-9]*)', out)[0])
        compress_ratio = float(re.findall('compression ratio = ([.0-9]*)', out)[0])
        compressed_size = float(re.findall('Encoded size = ([.0-9]*)', out)[0])
        errorx += float(re.findall('X cumulated error = ([.0-9]*)', out)[0])
        errory += float(re.findall('Y cumulated error = ([.0-9]*)', out)[0])
        errorz += float(re.findall('Z cumulated error = ([.0-9]*)', out)[0])
        max_diffx0 = float(re.findall('X Max absolute error = ([.0-9]*)', out)[0])
        max_diffy0 = float(re.findall('Y Max absolute error = ([.0-9]*)', out)[0])
        max_diffz0 = float(re.findall('Z Max absolute error = ([.0-9]*)', out)[0])
        max_diffx = max(max_diffx, max_diffx0)
        max_diffy = max(max_diffy, max_diffy0)
        max_diffz = max(max_diffz, max_diffz0)
        psnr0 = float(re.findall('overall PSNR = ([.0-9]*)', out)[0])

        exe = "{}/draco_encoder -point_cloud -i {} -o {} -qp {} 2>&1".format(
            draco_exe_folder, src_file, compressed_file, qp)
        out = os.popen(exe).read()
        compress_time = float(re.findall('compression time = ([.0-9]*)', out)[0])

        # compress_time = 0
        total_compress_time += compress_time
        total_compressed_size += compressed_size

        exe = "{}/draco_decoder -i {}  -o {} 2>&1".format(
            draco_exe_folder, compressed_file, decompressed_file)
        # print(exe)
        out = os.popen(exe).read()
        # print(out)
        decompress_time = float(re.findall('decompression time = ([.0-9]*)', out)[0])
        # print('decompress_time=', decompress_time)
        # decompress_time = 0
        total_decompress_time += decompress_time
        # dec_data[ts:ts + block] = np.fromfile(decompressed_file, dtype=np.float32).reshape(data_slice.shape)

        msg = 'time frame = {}, qp = {}, eb = {:.6f} compression_ratio = {:.3f}, psnr={:.3f} compress_time={:.6f}, decompress_time={:.6f}'.format(
            ts, qp, max(max_diffx0, max_diffy0, max_diffz0), compress_ratio, psnr0, total_compress_time, total_decompress_time)

        print(msg)
        # os.remove(src_file)
        os.remove(compressed_file)
        os.remove(decompressed_file)

    if len(tss) == 1:
        nrmsex = math.sqrt((errorx / r1)) / (np.max(x[tss[0]]) - np.min(x[tss[0]]))
        nrmsey = math.sqrt((errory / r1)) / (np.max(y[tss[0]]) - np.min(y[tss[0]]))
        nrmsez = math.sqrt((errorz / r1)) / (np.max(z[tss[0]]) - np.min(z[tss[0]]))
        nrmse = math.sqrt((nrmsex * nrmsex + nrmsey * nrmsey + nrmsez * nrmsez) / 3)
        psnr = -20 * math.log10(nrmse) if nrmse > 0 else 0
        ratio = r1 * 3 * 4.0 / total_compressed_size
        msg = 'method=draco, file = {}, qp = {}, timeframe = {} eb = {:.6f}, compression_ratio = {:.3f}, psnr={:.3f}, nrmse={:e}, compress_time={:.3f}, decompress_time={:.3f}'.format(
            Path(filenamex).absolute().parent.name, qp, tss[0], max(max_diffx, max_diffy, max_diffz), ratio, psnr,
            nrmse,
            total_compress_time,
            total_decompress_time)
    else:
        nrmsex = math.sqrt((errorx / r1 / timestep)) / (np.max(x) - np.min(x))
        nrmsey = math.sqrt((errory / r1 / timestep)) / (np.max(y) - np.min(y))
        nrmsez = math.sqrt((errorz / r1 / timestep)) / (np.max(z) - np.min(z))
        nrmse = math.sqrt((nrmsex * nrmsex + nrmsey * nrmsey + nrmsez * nrmsez) / 3)
        psnr = -20 * math.log10(nrmse) if nrmse > 0 else 0
        ratio = r1 * timestep * 3 * 4.0 / total_compressed_size
        msg = 'method=draco, file = {}, qp = {}, eb = {:.6f}, compression_ratio = {:.3f}, psnr={:.3f}, nrmse={:e}, compress_time={:.3f}, decompress_time={:.3f}'.format(
            Path(filenamex).absolute().parent.name, qp, max(max_diffx, max_diffy, max_diffz), ratio, psnr, nrmse,
            total_compress_time,
            total_decompress_time)

    print(msg)
