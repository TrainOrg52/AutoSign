class Vehicle:
    def __init__(self, id, timestamp, title, conformanceStatus):
        self.id = id
        self.timestamp = timestamp
        self.title = title
        self.conformanceStatus = conformanceStatus

    @staticmethod
    def from_doc(doc):
        data = doc.to_dict()

        return Vehicle(
            doc.id,
            data["timestamp"],
            data["title"],
            data["conformanceStatus"]
        )

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "title": self.title,
            "conformanceStatus": self.conformanceStatus
        }

    def update(self, db):
        db.collection(u'vehicles').document(self.id).set(self.to_dict())


class Walkthrough:
    def __init__(self, id, timestamp, checkpoints):
        self.id = id
        self.timestamp = timestamp
        self.checkpoints = checkpoints

    @staticmethod
    def from_doc(doc):
        data = doc.to_dict()

        return Walkthrough(
            doc.id,
            data["timestamp"],
            data["checkpoints"]
        )

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "checkpoints": self.checkpoints,
        }

    def update(self, db):
        db.collection(u'walkthroughs').document(self.id).set(self.to_dict())


class Checkpoint:
    def __init__(self, id, timestamp, vehicleID, title, prompt, captureType, signs):
        self.id = id
        self.timestamp = timestamp
        self.vehicleID = vehicleID
        self.title = title
        self.prompt = prompt
        self.captureType = captureType
        self.signs = signs

    @staticmethod
    def from_doc(doc):
        data = doc.to_dict()

        return Checkpoint(
            doc.id,
            data["timestamp"],
            data["vehicleID"],
            data["title"],
            data["prompt"],
            data["captureType"],
            data["signs"]
        )

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "vehicleID": self.vehicleID,
            "title": self.title,
            "prompt": self.prompt,
            "captureType": self.captureType,
            "signs": self.signs
        }

    def update(self, db):
        db.collection(u'checkpoints').document(self.id).set(self.to_dict())


class InspectionWalkthrough:
    def __init__(self, id, timestamp, vehicleID, processingStatus, conformanceStatus, checkpoints):
        self.id = id
        self.timestamp = timestamp
        self.vehicleID = vehicleID
        self.processingStatus = processingStatus
        self.conformanceStatus = conformanceStatus
        self.checkpoints = checkpoints

    @staticmethod
    def from_doc(doc):
        data = doc.to_dict()

        return InspectionWalkthrough(
            doc.id,
            data["timestamp"],
            data["vehicleID"],
            data["processingStatus"],
            data["conformanceStatus"],
            data["checkpoints"]
        )

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "vehicleID": self.vehicleID,
            "processingStatus": self.processingStatus,
            "conformanceStatus": self.conformanceStatus,
            "checkpoints": self.checkpoints
        }

    def update(self, db):
        db.collection(u'inspectionWalkthroughs').document(self.id).set(self.to_dict())


class InspectionCheckpoint:
    def __init__(self, id, timestamp, vehicleID, title, conformanceStatus, signs):
        self.id = id
        self.timestamp = timestamp
        self.vehicleID = vehicleID
        self.title = title
        self.conformanceStatus = conformanceStatus
        self.signs = signs

    @staticmethod
    def from_doc(doc):
        data = doc.to_dict()

        return InspectionCheckpoint(
            doc.id,
            data["timestamp"],
            data["vehicleID"],
            data["title"],
            data["conformanceStatus"],
            data["signs"]
        )

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "vehicleID": self.vehicleID,
            "title": self.title,
            "conformanceStatus": self.conformanceStatus,
            "signs": self.signs
        }

    def update(self, db):
        db.collection(u'inspectionCheckpoints').document(self.id).set(self.to_dict())
