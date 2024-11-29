# import cv2
# import face_recognition
# import asyncio
# import nest_asyncio
# from qdrant_client import QdrantClient
# from concurrent.futures import ThreadPoolExecutor, as_completed

# nest_asyncio.apply()  # Allow asyncio in environments with a running event loop

# # Connect to Qdrant instance
# client = QdrantClient(host="localhost", port=6334, prefer_grpc=True)
# collection_name = "embedding_collection1"

# # Compare embeddings against the stored ones in Qdrant
# def compare(embedding, top_k=1):
#     threshold = 0.93
#     search_results = client.search(
#         collection_name=collection_name,
#         query_vector=embedding,
#         limit=top_k
#     )
#     if not search_results:
#         return "Unknown"
#     result = search_results[0]
#     if result.score >= threshold:
#         return result.payload["name"]
#     else:
#         return "Unknown"

# # Predict face recognition for the current frame
# async def predict(frame):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_frame)
#     if not face_locations:
#         return []
    
#     face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
#     predictions = []

#     with ThreadPoolExecutor() as executor:
#         future_to_face = {
#             executor.submit(compare, face_encoding): face_location
#             for face_encoding, face_location in zip(face_encodings, face_locations)
#         }
#         for future in as_completed(future_to_face):
#             result = future.result()
#             location = future_to_face[future]
#             predictions.append((result, location))
    
#     return predictions

# # Display predictions on the webcam feed
# def show_predictions_on_frame(frame, predictions):
#     for name, (top, right, bottom, left) in predictions:
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

# # Process the webcam feed
# async def process_video():
#     video_capture = cv2.VideoCapture(0)
#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break
#         predictions = await predict(frame)
#         show_predictions_on_frame(frame, predictions)
#         cv2.imshow("Webcam", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     video_capture.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     asyncio.run(process_video())



# import cv2
# import face_recognition
# import asyncio
# import nest_asyncio
# from qdrant_client import QdrantClient
# from concurrent.futures import ThreadPoolExecutor, as_completed

# nest_asyncio.apply()  # Allow asyncio in environments with a running event loop

# # Connect to Qdrant instance
# client = QdrantClient(host="localhost", port=6334, prefer_grpc=True)
# collection_name = "embedding_collection1"

# # Compare embeddings against the stored ones in Qdrant
# def compare(embedding, top_k=1):
#     threshold = 0.93
#     search_results = client.search(
#         collection_name=collection_name,
#         query_vector=embedding,
#         limit=top_k
#     )
#     if not search_results:
#         return "Unknown"
#     result = search_results[0]
#     if result.score >= threshold:
#         return result.payload["name"]
#     else:
#         return "Unknown"

# # Process face detection and recognition in a separate thread
# def detect_and_recognize_faces(frame):
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_frame)
#     if not face_locations:
#         return []

#     face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
#     results = [(compare(encoding), location) for encoding, location in zip(face_encodings, face_locations)]
#     return results

# # Display predictions on the webcam feed
# def show_predictions_on_frame(frame, predictions):
#     for name, (top, right, bottom, left) in predictions:
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

# # Process the webcam feed asynchronously
# async def process_video():
#     video_capture = cv2.VideoCapture(0)
#     frame_skip = 1  # Process every 5th frame
#     frame_count = 0

#     with ThreadPoolExecutor(max_workers=16) as executor:
#         loop = asyncio.get_event_loop()
#         while True:
#             ret, frame = video_capture.read()
#             if not ret:
#                 break

#             frame_count += 1
#             if frame_count % frame_skip == 0:
#                 # Run face detection and recognition in a thread
#                 predictions = await loop.run_in_executor(executor, detect_and_recognize_faces, frame)
#                 show_predictions_on_frame(frame, predictions)

#             # Display the frame
#             cv2.imshow("Webcam", frame)

#             # Exit on 'q' key press
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     video_capture.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     try:
#         asyncio.run(process_video())
#     except KeyboardInterrupt:
#         print("\nExiting...")














