import cv2

class ObjectTracker:
    def __init__(self, tracker_type):
        self.tracker_type = tracker_type
        self.tracker = self._create_tracker()

    def _create_tracker(self):
        if self.tracker_type == 'BOOSTING':
            return cv2.legacy.TrackerBoosting_create()
        elif self.tracker_type == 'MIL':
            return cv2.TrackerMIL_create()
        elif self.tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif self.tracker_type == 'TLD':
            return cv2.legacy.TrackerTLD_create()
        elif self.tracker_type == 'MEDIANFLOW':
            return cv2.legacy.TrackerMedianFlow_create()
        elif self.tracker_type == 'MOSSE':
            return cv2.legacy.TrackerMOSSE_create()
        elif self.tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        else:
            raise ValueError(f"Tracker type {self.tracker_type} not supported.")

    def run(self):
        cap = cv2.VideoCapture(0)
        success, frame = cap.read()
        bbox = cv2.selectROI(frame, False)
        ret = self.tracker.init(frame, bbox)

        while True:
            success, frame = cap.read()
            if not success:
                break

            timer = cv2.getTickCount()
            ret, bbox = self.tracker.update(frame)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            if ret:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (50, 170, 50), 2, 1)
            else:
                cv2.putText(frame, "Tracking failure detected", (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            cv2.putText(frame, f"{self.tracker_type} Tracker", (100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            cv2.imshow("Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker_type = 'KCF'
    tracker_app = ObjectTracker(tracker_type)
    tracker_app.run()
