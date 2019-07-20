from pathlib import Path
import xml.etree.ElementTree as XTree


def get_items(root, type):
    ns = {"vc": "http://schemas.microsoft.com/developer/msbuild/2003"}
    items = []
    for node in root.findall('./vc:ItemGroup/vc:{}'.format(type), ns):
        item_path = node.attrib.get('Include', None)
        if item_path:
            items.append(item_path.replace('\\', '/'))
    return items


def generate_cmake(proj_file, cmake_file):
    proj = Path(proj_file)
    cmake = "cmake_minimum_required(VERSION 3.3)\n"

    cmake += f"project({proj.name})\n\n"
    cmake += "set(TARGET_NAME ${PROJECT_NAME})\n"

    root = XTree.parse(proj_file)

    # Qt
    mocs = get_items(root, 'MOC')
    uics = get_items(root, 'UIC')
    qrcs = get_items(root, 'QRC')

    if mocs:
        cmake += "TRN_MOC(moc_cpps\n  "
        cmake += "\n  ".join(mocs)
        cmake += "\n)\n"

    if uics:
        cmake += "set(uic_files\n  "
        cmake += "\n  ".join(uics)
        cmake += "\n)\nTRN_WRAP_UI(uic_headers ${uic_files})\n"

    has_qt = bool(mocs or uics or qrcs)

    # fglsl & binaries
    shaders = get_items('fglsl_source_processor')
    if shaders:
        cmake += "TRN_WRAP_FGLSL(emb_binaries\n"
        max_len = len(max(shaders, key=len)) + 3
        for sh in shaders:
            cmake += " {} {}\n".format(sh.ljust(max_len), Path(sh).stem + "_shader")
        cmake += ")\n"

    vars = []

    has_binaries = bool(shaders or vars)

    # CPP files
    cpps = get_items(root, 'ClCompile')
    need_pch = False

    for pch in ['stdafx.cpp', 'StdAfx.cpp']
        if pch in cpps:
            need_pch = True
            cpps.remove(pch)
            break

    if need_pch:
        cmake += "TRN_ADD_PRECOMPILED_HEADER(cpps \"stdafx.h\")"
    cmake += "set(cpps\n  "
    cmake += "\n  ".join(cpps)
    cmake += "\n)\n"

    Path(cmake_file).write_text(cmake)
