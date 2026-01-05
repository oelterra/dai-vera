# dai_vera/gui/components/clickfix.py

def reliable_release(widget, callback):
    """
    macOS-safe click handler.
    Runs AFTER the UI finishes the click event.
    Does NOT interfere with CustomTkinter internals.
    """
    def handler(_event=None):
        widget.after_idle(callback)

    widget.bind("<ButtonRelease-1>", handler, add="+")

# aliases so old code never breaks again
make_click_reliable = reliable_release
reliable_click = reliable_release
