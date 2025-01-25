import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
import os

# Load YOLOv8 model
model = YOLO('yolo11n.pt')

def get_center_position(x1, y1, x2, y2):
    """Calculate center position of bounding box."""
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y

def create_heatmap(player_positions, player_id):
    """Create a heatmap for a specific player using seaborn with a pitch background."""
    if not player_positions:
        st.warning(f"No position data available for player {player_id}")
        return

    # Convert positions to DataFrame
    df = pd.DataFrame(player_positions)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Load and plot the pitch background image
    pitch_image = plt.imread('pitch.jpg')
    ax.imshow(pitch_image, extent=(0, 1, 0, 1), alpha=0.8, aspect='auto')

    # Create heatmap using seaborn
    sns.kdeplot(data=df, x='x', y='y', cmap='YlOrRd', fill=True, ax=ax)

    # Set plot limits to match normalized coordinates
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Invert y-axis to match video coordinate system
    ax.invert_yaxis()

    # Set title and labels
    plt.title(f'Player {player_id} Position Heatmap')
    plt.xlabel('X Position (normalized)')
    plt.ylabel('Y Position (normalized)')

    return fig

def process_video(video_path, specific_player_id=None):
    """Process video, track players, and return processed video path."""
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create temporary output video file
    temp_dir = tempfile.mkdtemp()
    temp_output_path = os.path.join(temp_dir, "processed_video.mp4")

    # Define codec and create VideoWriter
    if os.name == 'nt':  # Windows
        fourcc = cv2.VideoWriter_fourcc(*'H264')
    else:  # Linux/Mac
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

    player_positions = {}
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Streamlit progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
        status_text.text(f"Processing frame {frame_count}/{total_frames}")

        # Detect and track objects
        results = model.track(frame, persist=True)

        # Create a copy of the frame for drawing
        display_frame = frame.copy()

        # Draw bounding boxes and collect player positions
        for result in results:
            for box in result.boxes:
                if box.id is not None:
                    tracker_id = int(box.id[0])
                    
                    # Skip if we're tracking a specific player and this isn't them
                    if specific_player_id is not None and tracker_id != specific_player_id:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Store normalized center position
                    center_x, center_y = get_center_position(x1, y1, x2, y2)
                    normalized_x = center_x / frame_width
                    normalized_y = center_y / frame_height

                    if tracker_id not in player_positions:
                        player_positions[tracker_id] = []

                    player_positions[tracker_id].append({'x': normalized_x, 'y': normalized_y})

                    # Draw bounding box and ID label
                    # Use different color for specific player if one is selected
                    color = (0, 0, 255) if specific_player_id and tracker_id == specific_player_id else (0, 255, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        display_frame, f"ID: {tracker_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                    )

        # Draw trail for specific player
        if specific_player_id and specific_player_id in player_positions:
            positions = player_positions[specific_player_id]
            if len(positions) > 1:
                # Draw last 30 positions as a trail
                trail_length = min(30, len(positions))
                for i in range(trail_length - 1):
                    start_pos = (
                        int(positions[-trail_length + i]['x'] * frame_width),
                        int(positions[-trail_length + i]['y'] * frame_height)
                    )
                    end_pos = (
                        int(positions[-trail_length + i + 1]['x'] * frame_width),
                        int(positions[-trail_length + i + 1]['y'] * frame_height)
                    )
                    # Fade the trail color based on position
                    alpha = (i + 1) / trail_length
                    color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                    cv2.line(display_frame, start_pos, end_pos, color, 2)

        out.write(display_frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    status_text.text("Video processing complete!")
    return player_positions, temp_output_path

def main():
    st.title("Player Tracking and Position Analysis")

    # Store the uploaded video path in session state
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None

    # Upload video file
    video_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])

    if video_file:
        # Create temporary input file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(video_file.read())
        st.session_state.video_path = temp_input.name
        temp_input.close()

        if st.button("Process Video"):
            # Process video and get player positions and output path
            player_positions, processed_video_path = process_video(st.session_state.video_path)

            # Display processed video
            st.header("Processed Video with Player Tracking")
            with open(processed_video_path, 'rb') as f:
                st.video(f.read())

            # Clean up temporary processed video
            os.remove(processed_video_path)

            # Store player positions in session state
            st.session_state.player_positions = player_positions

    # Check if player positions are available
    if 'player_positions' in st.session_state:
        player_positions = st.session_state.player_positions

        # Display player heatmap buttons
        st.header("Select Player to View Heatmap and Track")
        
        # Create columns for a better layout
        cols = st.columns(3)
        
        # Create buttons for each player
        for idx, player_id in enumerate(player_positions):
            col_idx = idx % 3
            with cols[col_idx]:
                if st.button(f"Track Player {player_id}", key=f"player_{player_id}"):
                    st.session_state.selected_player = player_id
                    
                    if st.session_state.video_path:
                        # Process video again for specific player
                        st.subheader(f"Tracking Player {player_id}")
                        player_positions, processed_video_path = process_video(
                            st.session_state.video_path, 
                            specific_player_id=player_id
                        )
                        
                        # Display processed video
                        with open(processed_video_path, 'rb') as f:
                            st.video(f.read())
                        
                        # Clean up temporary processed video
                        os.remove(processed_video_path)
                        
                        # Display heatmap
                        st.subheader(f"Player {player_id} Heatmap")
                        fig = create_heatmap(player_positions[player_id], player_id)
                        if fig:
                            st.pyplot(fig)

                        # Provide option to download player data
                        df = pd.DataFrame(player_positions[player_id])
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label=f"Download Player {player_id} Data",
                            data=csv,
                            file_name=f'player_{player_id}_positions.csv',
                            mime='text/csv'
                        )

    # Clean up temporary input file when the app is done
    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
        try:
            os.remove(st.session_state.video_path)
        except:
            pass

if __name__ == "__main__":
    main()
