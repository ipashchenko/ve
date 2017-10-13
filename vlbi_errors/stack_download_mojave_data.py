from mojave import download_mojave_uv_fits


save_dir = '/home/ilya/fs/sshfs/frb/data'
with open('/home/ilya/Dropbox/stack/sources', 'r') as fo:
    lines = fo.readlines()

lines = lines[1:]
sources = list()
for line in lines:
    sources.append(line.strip('\n').split(" ")[0])


for source in sources:
    download_mojave_uv_fits(source, bands=['u'], download_dir=save_dir)
