import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ----------------------------
# CONFIG
# ----------------------------
CATEGORY_COLOR = {
    "optimization": "#4CAF50",
    "architecture": "#2196F3",
    "training": "#FF9800",
    "theory": "#9C27B0",
    "default": "#9E9E9E"
}

# Box and text settings
BOX_WIDTH_MIN = 2.5
BOX_HEIGHT_MIN = 1.2
PADDING_X = 0.10
PADDING_Y = 0.05

# Font sizes
FONT_TITLE = 10
FONT_SUBTITLE = 7.5
FONT_TAKEAWAY = 7

# Spacing within boxes (in inches)
LINE_SPACING = 0.05
TITLE_SUB_SPACING = 0.00
SUB_DIV_SPACING = 0.02
DIV_TAKEAWAY_SPACING = 0.05

# Spacing (in inches) - reduced for tighter layout
YEAR_SPACING = 1.0
PAPER_SPACING = 0.5

# ----------------------------
# LOAD YAML
# ----------------------------
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------
# TEXT MEASUREMENT
# ----------------------------
def measure_text(ax, text, fontsize):
    t = ax.text(0, 0, text, fontsize=fontsize)
    fig = ax.get_figure()
    fig.canvas.draw()

    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    bbox_data = bbox.transformed(ax.transData.inverted())

    t.remove()

    return bbox_data.width, bbox_data.height

def estimate_box_size(paper):
    """Estimate box size by measuring actual text width."""
    # We'll need to measure in context, so return a function that needs ax
    title = f"{paper['date']} / {paper['alias']}"
    subtitle = paper["title"]
    takeaways = paper.get("takeaways", [])
    
    return {
        "title": title,
        "subtitle": subtitle,
        "takeaways": takeaways
    }

def line_height(fontsize):
    # stable heuristic instead of bbox measurement
    return fontsize * 0.008  # empirical for matplotlib

def compute_box_dimensions(ax, paper):
    title = f"{paper['date']} / {paper['alias']}"
    subtitle = paper["title"]
    takeaways = paper.get("takeaways", [])



    # --- measure title ---
    _, title_h = measure_text(ax, title, FONT_TITLE)

    # --- measure subtitle ---
    _, subtitle_h = measure_text(ax, subtitle, FONT_SUBTITLE)

    # --- measure takeaways individually ---
    takeaway_heights = []
    max_width = 0

    for t in takeaways:
        w, h = measure_text(ax, f"• {t['text']}", FONT_TAKEAWAY)
        takeaway_heights.append(h)
        max_width = max(max_width, w)

    # --- height (REAL stacking) ---
    height = (
        PADDING_Y * 2 +
        title_h +
        TITLE_SUB_SPACING +
        subtitle_h +
        SUB_DIV_SPACING +
        DIV_TAKEAWAY_SPACING +
        sum(takeaway_heights) +
        max(0, len(takeaways) - 1) * LINE_SPACING
    )

    # --- width ---
    title_w, _ = measure_text(ax, title, FONT_TITLE)
    subtitle_w, _ = measure_text(ax, subtitle, FONT_SUBTITLE)

    width = max(title_w, subtitle_w, max_width) + PADDING_X * 2

    return max(width, BOX_WIDTH_MIN), max(height, BOX_HEIGHT_MIN)*0.35

# ----------------------------
# DRAW PAPER BOX
# ----------------------------
def draw_paper(ax, paper, x, y):
    color = CATEGORY_COLOR.get(paper["category"], CATEGORY_COLOR["default"])
    
    box_width, box_height = compute_box_dimensions(ax, paper)

    box = FancyBboxPatch(
        (x, y),
        box_width,
        box_height,
        boxstyle="round,pad=0.06",
        edgecolor=color,
        facecolor="white",
        linewidth=1.8,
        alpha=0.95
    )
    ax.add_patch(box)
    cursor = y + box_height - PADDING_Y

    # fixed line system
    title_h = line_height(FONT_TITLE)
    subtitle_h = line_height(FONT_SUBTITLE)
    takeaway_h = line_height(FONT_TAKEAWAY)

    # --- TITLE ---
    title_text = f"{paper['date']} / {paper['alias']}"
    ax.text(x + PADDING_X, cursor, title_text,
            fontsize=FONT_TITLE, weight="bold", va="top")
    cursor -= title_h + TITLE_SUB_SPACING

    # --- SUBTITLE ---
    subtitle_text = paper["title"]
    ax.text(x + PADDING_X, cursor, subtitle_text,
            fontsize=FONT_SUBTITLE, color="#666", style="italic", va="top")
    cursor -= subtitle_h + SUB_DIV_SPACING

    # --- DIVIDER ---
    divider_y = cursor
    ax.plot(
        [x + PADDING_X, x + box_width - PADDING_X],
        [divider_y, divider_y],
        color=color,
        linewidth=1.2,
        alpha=0.6
    )
    cursor -= DIV_TAKEAWAY_SPACING

    # --- TAKEAWAYS ---
    for i, t in enumerate(paper.get("takeaways", [])):
        takeaway_text = f"• {t['text']}"
        ax.text(
            x + PADDING_X,
            cursor,
            takeaway_text,
            fontsize=FONT_TAKEAWAY,
            va="top"
        )
        cursor -= takeaway_h
        if i < len(paper["takeaways"]) - 1:
            cursor -= LINE_SPACING

    return box_width, box_height, divider_y


# ----------------------------
# LAYOUT ENGINE
# ----------------------------
def compute_positions(ax, papers):
    """Compute positions using real box sizes (width-aware & height-aware)."""
    
    # ----------------------------
    # STEP 1: Measure all boxes
    # ----------------------------
    paper_sizes = {}
    for p in papers:
        w, h = compute_box_dimensions(ax, p)
        paper_sizes[p["id"]] = (w, h)
    
    # ----------------------------
    # STEP 2: Group by year
    # ----------------------------
    groups = {}
    for p in papers:
        year = p["date"][:2]
        groups.setdefault(year, []).append(p)
    
    # Sort years
    years = sorted(groups.keys())
    
    # ----------------------------
    # STEP 3: Compute column widths
    # ----------------------------
    year_widths = {}
    for year, ps in groups.items():
        year_widths[year] = max(paper_sizes[p["id"]][0] for p in ps)
    
    # ----------------------------
    # STEP 4: Assign X positions (cumulative)
    # ----------------------------    
    year_x = {}
    current_x = 0
    
    for year in years:
        year_x[year] = current_x
        current_x += year_widths[year] + YEAR_SPACING
    
    # ----------------------------
    # STEP 5: Assign Y positions (stack by height)
    # ----------------------------  
    pos = {}
    
    for year, ps in groups.items():
        # Sort descending by date
        ps.sort(key=lambda x: x["date"], reverse=True)
        
        y_cursor = 0
        
        for p in ps:
            w, h = paper_sizes[p["id"]]
            
            pos[p["id"]] = (year_x[year], -y_cursor)
            
            y_cursor += h + PAPER_SPACING
    
    return pos, paper_sizes


# ----------------------------
# GET ANCHOR POINTS
# ----------------------------
def get_takeaway_anchor(x, y, box_w, box_h, takeaway_index, divider_y):
    """Get the anchor point for an arrow from a takeaway."""
    # Estimate y position of the takeaway
    offset_from_divider = (takeaway_index + 1) * 0.25
    takeaway_y = divider_y - offset_from_divider
    
    # Return right edge of box at takeaway height
    return (x + box_w, takeaway_y)


def get_paper_title_anchor(x, y, box_h):
    """Get the anchor point at the title/center of a paper."""
    return (x, y + box_h * 0.65)


# ----------------------------
# DRAW ARROW WITH CURVE
# ----------------------------
def draw_arrow(ax, start, end, color):
    """Draw a curved arrow from start to end."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=20,
        color=color,
        linewidth=1.8,
        alpha=0.7,
        connectionstyle="arc3,rad=0.3",
        zorder=1
    )
    ax.add_patch(arrow)



# ----------------------------
# MAIN RENDER
# ----------------------------
def render(data, output_path="paper_map.png"):
    """Generate the paper relationship graph."""
    papers = data.get("papers", [])
    relations = data.get("relations", [])
    
    if not papers:
        print("❌ No papers found in YAML")
        return
    
    print(f"✓ Loaded {len(papers)} papers, {len(relations)} relations")
        
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 14), dpi=100)
    ax.set_facecolor("#f9f9f9")
    ax.axis("off")
    
    # Compute positions
    pos, paper_sizes = compute_positions(ax, papers)

    # Store paper info (size, divider position)
    paper_info = {}
    
    # Draw all papers first
    for p in papers:
        x, y = pos[p["id"]]
        w, h, divider_y = draw_paper(ax, p, x, y)
        paper_info[p["id"]] = {
            "pos": (x, y),
            "width": w,
            "height": h,
            "divider_y": divider_y
        }
    
    # Draw relations
    paper_map = {p["id"]: p for p in papers}
    
    for r in relations:
        try:
            parts = r["from"].split(".")
            if len(parts) != 2:
                print(f"⚠ Invalid relation format: {r['from']}")
                continue
            
            from_id, takeaway_id = parts
            to_id = r["to"]
            
            if from_id not in paper_map:
                print(f"⚠ Paper '{from_id}' not found")
                continue
            if to_id not in paper_map:
                print(f"⚠ Paper '{to_id}' not found")
                continue
            
            # Find takeaway index
            from_paper = paper_map[from_id]
            takeaway_idx = -1
            for i, t in enumerate(from_paper.get("takeaways", [])):
                if t["id"] == takeaway_id:
                    takeaway_idx = i
                    break
            
            if takeaway_idx == -1:
                print(f"⚠ Takeaway '{takeaway_id}' not found in '{from_id}'")
                continue
            
            # Get positions
            from_info = paper_info[from_id]
            to_info = paper_info[to_id]
            
            x1, y1 = from_info["pos"]
            x2, y2 = to_info["pos"]
            
            # Get anchor points
            start = get_takeaway_anchor(
                x1, y1,
                from_info["width"],
                from_info["height"],
                takeaway_idx,
                from_info["divider_y"]
            )
            
            end = get_paper_title_anchor(x2, y2, to_info["height"])
            
            # Get color
            color = CATEGORY_COLOR.get(from_paper["category"], CATEGORY_COLOR["default"])
            
            # Draw arrow
            draw_arrow(ax, start, end, color)
            
        except Exception as e:
            print(f"❌ Error drawing relation: {e}")
    
    # Set axis limits with padding
    all_x = [paper_info[p["id"]]["pos"][0] + paper_info[p["id"]]["width"] for p in papers]
    all_y_bottom = [paper_info[p["id"]]["pos"][1] for p in papers]
    all_y_top = [paper_info[p["id"]]["pos"][1] + paper_info[p["id"]]["height"] for p in papers]
    
    ax.set_xlim(min(paper_info[p["id"]]["pos"][0] for p in papers) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y_bottom) - 1, max(all_y_top) + 1)
    
    # Save
    try:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor="#f9f9f9")
        print(f"✓ Graph saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving: {e}")
        return
    
    # plt.show()



# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    import sys
    import os
    
    yaml_path = "./paper map/papers.yaml"
    output_path = "paper map.png"
    
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    if not os.path.exists(yaml_path):
        print(f"❌ File not found: {yaml_path}")
        sys.exit(1)
    
    try:
        data = load_yaml(yaml_path)
        render(data, output_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)