#!/usr/bin/env python3
"""Teil A: Foto-Faceswap mit FLUX.2 Klein 9B + BFS-LoRA über die ComfyUI-API.

Du baust den Workflow einmal in ComfyUI (Klein 9B + BFS-LoRA + ggf. Flux2Klein-Enhancer-Nodes),
gibst diesen Nodes die Titel (Rechtsklick -> Title):
    SOURCE  = Load Image mit dem Gesicht, das eingesetzt wird
    TARGET  = Load Image mit dem Bild, in dem getauscht wird
    OUTPUT  = Save Image
    PROMPT  = (optional) Text-Encode-Node mit dem Prompt
und exportierst ihn per "Export (API)" als workflow_api.json.

    python comfy_faceswap.py --workflow workflow_api.json --source gesicht.jpg --target bild.jpg --out ergebnis.png
"""
import argparse, copy, json, os, random, sys, time, uuid
import requests


def find(wf, title):
    return [k for k, n in wf.items() if n.get("_meta", {}).get("title", "").strip().upper() == title]


def upload(server, path):
    with open(path, "rb") as f:
        r = requests.post(f"{server}/upload/image", files={"image": (os.path.basename(path), f)}, data={"overwrite": "true"})
    r.raise_for_status()
    j = r.json()
    return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]


def swap(workflow, source, target, out, server="http://127.0.0.1:8188", prompt=None, seed=None, timeout=1800):
    wf = copy.deepcopy(json.load(open(workflow, encoding="utf-8")) if isinstance(workflow, str) else workflow)
    for title in ("SOURCE", "TARGET", "OUTPUT"):
        if not find(wf, title):
            sys.exit(f"Im Workflow fehlt ein Node mit dem Titel '{title}' (siehe Hilfe oben).")
    for k in find(wf, "SOURCE"):
        wf[k]["inputs"]["image"] = upload(server, source)
    for k in find(wf, "TARGET"):
        wf[k]["inputs"]["image"] = upload(server, target)
    if prompt is not None:
        for k in find(wf, "PROMPT"):
            key = "text" if "text" in wf[k]["inputs"] else next(iter(wf[k]["inputs"]))
            wf[k]["inputs"][key] = prompt
    seed = random.randint(0, 2**32 - 1) if seed is None else seed
    for n in wf.values():
        for key in ("seed", "noise_seed"):
            if isinstance(n.get("inputs", {}).get(key), int):
                n["inputs"][key] = seed

    r = requests.post(f"{server}/prompt", json={"prompt": wf, "client_id": str(uuid.uuid4())})
    if r.status_code != 200:
        sys.exit(f"ComfyUI hat den Workflow abgelehnt:\n{r.text}")
    pid = r.json()["prompt_id"]
    out_nodes = find(wf, "OUTPUT")
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = requests.get(f"{server}/history/{pid}").json()
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                sys.exit(f"Fehler in ComfyUI: {json.dumps(st.get('messages', []))[:2000]}")
            for k in out_nodes:
                imgs = h[pid].get("outputs", {}).get(k, {}).get("images", [])
                if imgs:
                    im = imgs[0]
                    data = requests.get(f"{server}/view", params={"filename": im["filename"], "subfolder": im.get("subfolder", ""),
                                                                  "type": im.get("type", "output")}).content
                    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
                    open(out, "wb").write(data)
                    return out
            if st.get("completed"):
                sys.exit("Workflow fertig, aber der OUTPUT-Node hat kein Bild geliefert.")
        time.sleep(1)
    sys.exit("Zeitüberschreitung beim Warten auf ComfyUI.")


def main():
    p = argparse.ArgumentParser(description="Foto-Faceswap über ComfyUI (Klein 9B + BFS)")
    p.add_argument("--workflow", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", default="ergebnis.png")
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--prompt", help="ersetzt den Text im PROMPT-Node (z. B. Gesichtsausdruck beschreiben)")
    p.add_argument("--seed", type=int)
    a = p.parse_args()
    print("Fertig:", swap(a.workflow, a.source, a.target, a.out, a.server, a.prompt, a.seed))


if __name__ == "__main__":
    main()
