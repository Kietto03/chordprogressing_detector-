"""
Visualization helpers for the Streamlit UI (PR4 refactor).
Self-contained, no cross-origin / parent DOM access.
"""

import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Any


ROOT_COLORS = {
    'C': '#EF4444', 'C#': '#F97316', 'D': '#F59E0B', 'D#': '#EAB308',
    'E': '#10B981', 'F': '#14B8A6', 'F#': '#06B6D4', 'G': '#3B82F6',
    'G#': '#6366F1', 'A': '#8B5CF6', 'A#': '#A855F7', 'B': '#EC4899',
    'N': '#94A3B8'
}


def build_chord_timeline_figure(
    y: np.ndarray,
    sr: int,
    df_chords: "pd.DataFrame",
    scrub_time: float = 0.0,
    downsample: int = 4,
) -> go.Figure:
    """
    Build an interactive Plotly timeline:
    - Downsampled waveform / envelope
    - Colored horizontal rectangles for each chord segment (by root)
    - Vertical marker line at scrub_time
    - Hover shows chord + roman + time
    """
    import pandas as pd  # local to avoid top-level dep issues in some contexts

    fig = go.Figure()

    # Downsampled waveform (simple envelope for speed)
    if len(y) > 0:
        step = max(1, len(y) // (sr * 30 // downsample))  # ~30s visual at most
        y_ds = y[::step]
        t_ds = np.linspace(0, len(y) / sr, len(y_ds))
        fig.add_trace(go.Scatter(
            x=t_ds,
            y=y_ds,
            mode="lines",
            name="Waveform",
            line=dict(color="#64748B", width=0.8),
            opacity=0.6,
        ))

    total_duration = len(y) / sr if len(y) > 0 else 1.0

    # Chord segments as shapes (rects)
    shapes = []
    annotations = []
    if not df_chords.empty:
        for _, row in df_chords.iterrows():
            start = float(row["Start Time (s)"])
            end = float(row["End Time (s)"])
            chord = str(row.get("Chord_Clean", row.get("Chord", "N")))
            roman = str(row.get("Roman", ""))

            root = chord.split(":")[0] if ":" in chord else chord
            root = root.replace("min", "").replace("maj", "").replace("dim", "").replace("aug", "").replace("7", "").strip()
            color = ROOT_COLORS.get(root, "#94A3B8")

            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=start, x1=end,
                y0=0.05, y1=0.95,
                fillcolor=color,
                opacity=0.25,
                layer="below",
                line_width=0,
            ))

            # Small label on top
            mid = (start + end) / 2
            if end - start > 0.4:
                annotations.append(dict(
                    x=mid, y=0.92,
                    xref="x", yref="paper",
                    text=f"<b>{chord}</b>{' ' + roman if roman else ''}",
                    showarrow=False,
                    font=dict(size=10, color="#1E293B"),
                    align="center",
                ))

    # Current playhead / scrub marker
    if scrub_time is not None:
        shapes.append(dict(
            type="line",
            xref="x", yref="paper",
            x0=scrub_time, x1=scrub_time,
            y0=0, y1=1,
            line=dict(color="#7928CA", width=3, dash="solid"),
            layer="above",
        ))

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(
            title="Time (s)",
            range=[0, total_duration],
            showgrid=True,
            gridcolor="#E2E8F0",
        ),
        yaxis=dict(visible=False),
        height=220,
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False,
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#F8FAFC",
        hovermode="x unified",
    )

    # Add a dummy trace for hover info on segments (optional polish)
    if not df_chords.empty:
        fig.add_trace(go.Scatter(
            x=df_chords["Start Time (s)"],
            y=[0.5] * len(df_chords),
            mode="markers",
            marker=dict(size=0, color="rgba(0,0,0,0)"),
            hovertext=[
                f"{row.get('Chord_Clean', row.get('Chord'))} {row.get('Roman','')}<br>"
                f"{row['Start Time (s)']:.2f}s – {row['End Time (s)']:.2f}s"
                for _, row in df_chords.iterrows()
            ],
            hoverinfo="text",
            showlegend=False,
        ))

    return fig
