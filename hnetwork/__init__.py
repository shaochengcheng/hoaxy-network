from pkg_resources import resource_filename

version_file = resource_filename('hnetwork', 'VERSION')

with open(version_file, 'r') as f:
    VERSION = f.read().strip()
