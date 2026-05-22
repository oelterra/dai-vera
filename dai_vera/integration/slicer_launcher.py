from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile

import nibabel as nib
import numpy as np


def normalize_slicer_executable(path: str) -> str:
    if not path:
        return ""

    expanded = os.path.abspath(os.path.expanduser(path))
    if platform.system() == "Darwin" and expanded.endswith(".app"):
        candidate = os.path.join(expanded, "Contents", "MacOS", "Slicer")
        if os.path.isfile(candidate):
            return candidate
    return expanded


def find_slicer_executable(configured_path: str = "") -> str | None:
    normalized = normalize_slicer_executable(configured_path)
    if normalized and os.path.exists(normalized):
        return normalized

    executable_names = ["Slicer", "Slicer.exe"]
    for name in executable_names:
        located = shutil.which(name)
        if located:
            return located

    system_name = platform.system()
    common_locations: list[str] = []
    if system_name == "Darwin":
        common_locations = [
            "/Applications/Slicer.app",
            os.path.expanduser("~/Applications/Slicer.app"),
        ]
    elif system_name == "Windows":
        common_locations = [
            r"C:\Program Files\Slicer 5.8.1\Slicer.exe",
            r"C:\Program Files\Slicer 5.6.2\Slicer.exe",
            r"C:\Program Files\Slicer 5.6.1\Slicer.exe",
            r"C:\Program Files\Slicer.org\Slicer 5.8.1\Slicer.exe",
            r"C:\Program Files\Slicer.org\Slicer 5.6.2\Slicer.exe",
        ]

    for location in common_locations:
        normalized_location = normalize_slicer_executable(location)
        if os.path.exists(normalized_location):
            return normalized_location

    return None


def find_totalsegmentator_executable() -> str | None:
    direct_cli = shutil.which("TotalSegmentator")
    if direct_cli:
        return direct_cli
    return None


def _write_cta_volume_nifti(cta_volume: dict, output_path: str) -> None:
    pixels = np.asarray(cta_volume.get("pixels"))
    if pixels.ndim != 4:
        raise RuntimeError(f"CTA volume is expected to be 4D, got shape {pixels.shape}.")
    if pixels.shape[0] < 1:
        raise RuntimeError("CTA volume does not contain any time points to export.")

    volume_3d = np.asarray(pixels[0], dtype=np.float32)
    nifti_data = np.transpose(volume_3d, (1, 2, 0))
    nib.save(nib.Nifti1Image(nifti_data, affine=np.eye(4, dtype=np.float32)), output_path)


def _generate_vessel_workflow_scaffold(
    cta_folder: str,
    fallback_nifti_path: str,
    handoff_dir: str,
    totalsegmentator_executable: str | None,
) -> str:
    return f"""# DAI Vera Vessel Analysis Scaffold for 3D Slicer
# Generated automatically when CTA is opened from DAI Vera.
#
# Suggested workflow:
# 1. Confirm the CTA volume is loaded in Slicer.
# 2. Run TotalSegmentator / vessel segmentation here if needed.
# 3. Apply your lumen-straightening ML pipeline.
# 4. Save the straightened lumen result to disk.
# 5. Import that result back into DAI Vera for lesion point placement and measurements.

import os
import shutil
import qt
import slicer
import subprocess
import sys

CTA_DICOM_FOLDER = r\"{cta_folder}\"
CTA_FALLBACK_NIFTI = r\"{fallback_nifti_path}\"
DAI_VERA_HANDOFF_DIR = r\"{handoff_dir}\"
TOTALSEG_OUTPUT_DIR = os.path.join(DAI_VERA_HANDOFF_DIR, "totalseg_output")
CORONARY_MASK_PATH = os.path.join(TOTALSEG_OUTPUT_DIR, "coronary_arteries.nii.gz")
TOTALSEG_EXECUTABLE = {totalsegmentator_executable!r}


def get_cta_volume_node():
    volumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    if not volumes:
        raise RuntimeError("No scalar volume node is loaded in Slicer.")
    return volumes[-1]


def resolve_totalsegmentator_command():
    if TOTALSEG_EXECUTABLE and os.path.isfile(TOTALSEG_EXECUTABLE):
        return [TOTALSEG_EXECUTABLE]
    direct_cli = shutil.which("TotalSegmentator")
    if direct_cli:
        return [direct_cli]
    return [sys.executable, "-m", "totalsegmentator"]


def build_clean_totalsegmentator_env(command):
    env = os.environ.copy()
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONEXECUTABLE",
        "PYTHONNOUSERSITE",
        "PYTHONSTARTUP",
    ):
        env.pop(key, None)

    executable_dir = os.path.dirname(command[0]) if command else ""
    if executable_dir:
        existing_path = env.get("PATH", "")
        env["PATH"] = executable_dir if not existing_path else executable_dir + os.pathsep + existing_path

    return env


def run_totalsegmentator_coronary(use_dicom_folder=True):
    \"\"\"Run coronary segmentation from Slicer using TotalSegmentator.\"\"\"
    volume_node = get_cta_volume_node()
    os.makedirs(TOTALSEG_OUTPUT_DIR, exist_ok=True)

    input_path = CTA_DICOM_FOLDER if use_dicom_folder else CTA_FALLBACK_NIFTI
    command = resolve_totalsegmentator_command() + [
        "-i",
        input_path,
        "-o",
        TOTALSEG_OUTPUT_DIR,
        "-ta",
        "coronary_arteries",
    ]
    print(f"Running TotalSegmentator for {{volume_node.GetName()}}")
    print("Command:", " ".join(command))
    env = build_clean_totalsegmentator_env(command)

    result = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("TotalSegmentator failed. See console output above.")

    print(f"Coronary segmentation complete: {{CORONARY_MASK_PATH}}")
    return CORONARY_MASK_PATH


def load_coronary_mask():
    if not os.path.isfile(CORONARY_MASK_PATH):
        raise RuntimeError("Coronary mask file was not found. Run run_totalsegmentator_coronary() first.")
    loaded = slicer.util.loadVolume(CORONARY_MASK_PATH)
    if not loaded:
        raise RuntimeError("Slicer could not load the coronary mask output.")
    print(f"Loaded coronary mask: {{CORONARY_MASK_PATH}}")
    return loaded


def straighten_lumen_placeholder():
    \"\"\"Replace this with your lumen-straightening algorithm call.\"\"\"
    volume_node = get_cta_volume_node()
    print(f"CTA ready for lumen straightening: {{volume_node.GetName()}}")
    print("TODO: run the ML algorithm that generates the straightened lumen view.")


def export_for_dai_vera(output_path):
    \"\"\"Save the straightened lumen output where DAI Vera can import it.\"\"\"
    if not output_path:
        raise ValueError("Provide an output file path for the straightened lumen result.")
    print(f"TODO: save the straightened lumen result to {{output_path}}")


if __name__ == "__main__":
    print("DAI Vera Slicer scaffold loaded.")
    print(f"CTA DICOM folder: {{CTA_DICOM_FOLDER}}")
    print(f"Fallback CTA NIfTI: {{CTA_FALLBACK_NIFTI}}")
    print(f"Handoff directory: {{DAI_VERA_HANDOFF_DIR}}")
    print("Next step in ScriptEditor: run_totalsegmentator_coronary()")
"""


def _build_slicer_startup_script(
    script_path: str,
    cta_folder: str,
    fallback_nifti_path: str,
    scaffold_path: str,
    handoff_dir: str,
    totalsegmentator_executable: str | None,
) -> None:
    scaffold_text = _generate_vessel_workflow_scaffold(
        cta_folder,
        fallback_nifti_path,
        handoff_dir,
        totalsegmentator_executable,
    )
    script = f"""
import os
import qt
import slicer
import DICOMLib.DICOMUtils as DICOMUtils

cta_folder = {cta_folder!r}
fallback_nifti = {fallback_nifti_path!r}
scaffold_text = {scaffold_text!r}
scaffold_path = {scaffold_path!r}


def load_from_dicom():
    loaded_volume = None
    with DICOMUtils.TemporaryDICOMDatabase() as dicom_db:
        DICOMUtils.importDicom(cta_folder, dicom_db)
        patient_uids = dicom_db.patients()
        if not patient_uids:
            raise RuntimeError("No importable DICOM patients were found in the CTA folder.")

        before_ids = set(node.GetID() for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"))
        for patient_uid in patient_uids:
            DICOMUtils.loadPatientByUID(patient_uid)
        after_nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")

        for node in reversed(after_nodes):
            if node.GetID() not in before_ids:
                loaded_volume = node
                break

    if loaded_volume is None:
        raise RuntimeError("CTA DICOMs were imported but no scalar volume node was loaded.")
    return loaded_volume


def load_from_nifti():
    if not os.path.isfile(fallback_nifti):
        raise RuntimeError("Fallback CTA NIfTI file was not created.")
    loaded = slicer.util.loadVolume(fallback_nifti)
    if not loaded:
        raise RuntimeError("Slicer could not load the fallback CTA NIfTI file.")
    return loaded if hasattr(loaded, "GetID") else slicer.mrmlScene.GetFirstNodeByName("DAI_Vera_CTA")


def add_scaffold_text_node():
    text_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode", "DAI_Vera_VesselWorkflow.py")
    text_node.SetAttribute("mimetype", "text/x-python")
    text_node.SetAttribute("customTag", "pythonFile")
    text_node.SetAttribute("TextNodeType", "Python")
    text_node.SetAttribute("TextType", "PythonScript")
    text_node.SetAttribute("ModuleName", "ScriptEditor")
    text_node.SetText(scaffold_text)
    storage_node = text_node.GetStorageNode()
    if storage_node is None:
        storage_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextStorageNode")
        text_node.SetAndObserveStorageNodeID(storage_node.GetID())
    if hasattr(storage_node, "SetSupportedReadFileExtensions"):
        storage_node.SetSupportedReadFileExtensions(["py"])
        storage_node.SetSupportedWriteFileExtensions(["py"])
    if storage_node is None:
        text_node.AddDefaultStorageNode()
        storage_node = text_node.GetStorageNode()
    if storage_node is not None:
        storage_node.SetFileName(scaffold_path)
        text_node.SetSaveWithScene(False)
        storage_node.WriteData(text_node)
    return text_node


def try_open_scaffold_in_script_editor(text_node):
    if not hasattr(slicer.modules, "scripteditor"):
        return False

    try:
        slicer.util.selectModule("ScriptEditor")
        slicer.app.processEvents()
        widget = slicer.util.getModuleWidget("ScriptEditor")
        if widget is None:
            return False

        if hasattr(widget, "setCurrentNode"):
            try:
                widget.setCurrentNode(text_node)
                slicer.app.processEvents()
                return True
            except Exception:
                pass

        candidate_selectors = []
        for child in slicer.util.findChildren(widget):
            if hasattr(child, "setCurrentNode"):
                candidate_selectors.append(child)

        for selector in candidate_selectors:
            try:
                selector.setCurrentNode(text_node)
                slicer.app.processEvents()
                return True
            except Exception:
                continue

        if hasattr(widget, "setEditedNode"):
            try:
                widget.setEditedNode(text_node)
                slicer.app.processEvents()
                return True
            except Exception:
                pass
    except Exception as editor_error:
        print(f"DAI Vera: could not auto-open ScriptEditor scaffold. {{editor_error}}")

    return False


def schedule_script_editor_selection_retry(text_node, attempts=6, delay_ms=900):
    if attempts <= 0:
        print("DAI Vera: ScriptEditor scaffold retry exhausted.")
        return

    def _retry():
        if try_open_scaffold_in_script_editor(text_node):
            print("DAI Vera: ScriptEditor scaffold node was selected on delayed retry.")
            return
        schedule_script_editor_selection_retry(text_node, attempts=attempts - 1, delay_ms=delay_ms)

    qt.QTimer.singleShot(delay_ms, _retry)


try:
    try:
        volume_node = load_from_dicom()
        print("DAI Vera: CTA loaded into 3D Slicer from DICOM folder.")
    except Exception as dicom_error:
        print(f"DAI Vera: DICOM import failed, falling back to NIfTI. {{dicom_error}}")
        volume_node = load_from_nifti()
        if volume_node is None:
            volume_node = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        print("DAI Vera: CTA loaded into 3D Slicer from fallback NIfTI.")

    if volume_node is None:
        raise RuntimeError("No CTA volume node was available after loading.")

    scaffold_node = add_scaffold_text_node()
    slicer.util.setSliceViewerLayers(background=volume_node)
    slicer.util.resetSliceViews()
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    print(f"DAI Vera: vessel workflow scaffold saved to {{scaffold_path}}")
    print(f"DAI Vera: scaffold text node created as {{scaffold_node.GetName()}}")
    if hasattr(slicer.modules, "scripteditor"):
        if try_open_scaffold_in_script_editor(scaffold_node):
            print("DAI Vera: ScriptEditor scaffold node was selected automatically.")
        else:
            print("DAI Vera: ScriptEditor detected, but the scaffold node could not be auto-selected.")
            schedule_script_editor_selection_retry(scaffold_node)
except Exception as exc:
    slicer.util.errorDisplay(f"DAI Vera failed to load CTA in 3D Slicer: {{exc}}")
    raise
"""
    with open(script_path, "w", encoding="utf-8") as handle:
        handle.write(script)


def launch_slicer_with_cta(
    slicer_executable: str,
    cta_folder: str,
    cta_volume: dict,
) -> subprocess.Popen:
    normalized_executable = normalize_slicer_executable(slicer_executable)
    if not normalized_executable or not os.path.exists(normalized_executable):
        raise RuntimeError("3D Slicer executable was not found. Choose a valid Slicer application first.")
    if not cta_folder or not os.path.isdir(cta_folder):
        raise RuntimeError("CTA folder not found. Load CTA images before opening 3D Slicer.")
    if not cta_volume:
        raise RuntimeError("CTA volume is not loaded in DAI Vera yet.")

    handoff_dir = tempfile.mkdtemp(prefix="dai_vera_slicer_")
    fallback_nifti_path = os.path.join(handoff_dir, "cta_for_slicer.nii.gz")
    script_path = os.path.join(handoff_dir, "load_cta_in_slicer.py")
    scaffold_path = os.path.join(handoff_dir, "vessel_workflow_scaffold.py")
    totalsegmentator_executable = find_totalsegmentator_executable()

    _write_cta_volume_nifti(cta_volume, fallback_nifti_path)
    scaffold_text = _generate_vessel_workflow_scaffold(
        cta_folder,
        fallback_nifti_path,
        handoff_dir,
        totalsegmentator_executable,
    )
    with open(scaffold_path, "w", encoding="utf-8") as handle:
        handle.write(scaffold_text)
    _build_slicer_startup_script(
        script_path,
        cta_folder,
        fallback_nifti_path,
        scaffold_path,
        handoff_dir,
        totalsegmentator_executable,
    )

    command = [normalized_executable, "--python-script", script_path]
    try:
        return subprocess.Popen(command)
    except Exception as exc:
        raise RuntimeError(f"Failed to open 3D Slicer: {exc}") from exc
