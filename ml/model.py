class Vehicle:
    def __init__(self, id, timestamp, title, conformanceStatus, location, checkpoints, lastVehicleInspectionID):
        self.id = id
        self.timestamp = timestamp
        self.title = title
        self.conformanceStatus = conformanceStatus
        self.location = location
        self.checkpoints = checkpoints
        self.lastVehicleInspectionID = lastVehicleInspectionID

    @staticmethod
    def from_doc(doc):
        data = doc.to_dict()

        return Vehicle(
            doc.id,
            data["timestamp"],
            data["title"],
            data["conformanceStatus"],
            data["location"],
            data["checkpoints"],
            data["lastVehicleInspectionID"]
        )

    def to_dict(self):
        return {
            "location": self.location,
            "timestamp": self.timestamp,
            "title": self.title,
            "conformanceStatus": self.conformanceStatus,
            "checkpoints": self.checkpoints,
            "lastVehicleInspectionID": self.lastVehicleInspectionID
        }

    def update(self, db):
        db.collection(u'vehicles').document(self.id).set(self.to_dict())


class Checkpoint:
    def __init__(self, id, timestamp, vehicleID, title, prompt, signs, conformanceStatus, lastVehicleInspectionID, lastVehicleInspectionResult, lastVehicleRemediationID):
        self.id = id
        self.timestamp = timestamp
        self.vehicleID = vehicleID
        self.title = title
        self.prompt = prompt
        self.signs = signs
        self.conformanceStatus = conformanceStatus
        self.lastVehicleInspectionID = lastVehicleInspectionID
        self.lastVehicleInspectionResult = lastVehicleInspectionResult
        self.lastVehicleRemediationID = lastVehicleRemediationID

    @staticmethod
    def from_doc(doc):
        data = doc.to_dict()

        return Checkpoint(
            doc.id,
            data["timestamp"],
            data["vehicleID"],
            data["title"],
            data["prompt"],
            data["signs"],
            data["conformanceStatus"],
            data["lastVehicleInspectionID"],
            data["lastVehicleInspectionResult"],
            data["lastVehicleRemediationID"],
        )

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "vehicleID": self.vehicleID,
            "title": self.title,
            "prompt": self.prompt,
            "conformanceStatus": self.conformanceStatus,
            "lastVehicleInspectionID": self.lastVehicleInspectionID,
            "lastVehicleInspectionResult": self.lastVehicleInspectionResult,
            "lastVehicleRemediationID": self.lastVehicleRemediationID,
            "signs": self.signs
        }

    def update(self, db):
        db.collection(u'checkpoints').document(self.id).set(self.to_dict())


class vehicleInspection:
    def __init__(self, id, timestamp, vehicleID, conformanceStatus, checkpoints):
        self.id = id
        self.timestamp = timestamp
        self.vehicleID = vehicleID
        self.conformanceStatus = conformanceStatus
        self.checkpoints = checkpoints

    @staticmethod
    def from_doc(doc):
        data = doc.to_dict()

        return vehicleInspection(
            doc.id,
            data["timestamp"],
            data["vehicleID"],
            data["conformanceStatus"],
            data["checkpoints"]
        )

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "vehicleID": self.vehicleID,
            "conformanceStatus": self.conformanceStatus,
            "checkpoints": self.checkpoints
        }

    def update(self, db):
        db.collection(u'vehicleInspections').document(self.id).set(self.to_dict())


class CheckpointInspection:
    def __init__(self, id, timestamp, vehicleID, title, conformanceStatus, signs, checkpointID):
        self.id = id
        self.timestamp = timestamp
        self.vehicleID = vehicleID
        self.checkpointID = checkpointID
        self.title = title
        self.conformanceStatus = conformanceStatus
        self.signs = signs

    @staticmethod
    def from_doc(doc):
        data = doc.to_dict()

        return CheckpointInspection(
            doc.id,
            data["timestamp"],
            data["vehicleID"],
            data["title"],
            data["conformanceStatus"],
            data["signs"],
            data["checkpointID"]
        )

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "vehicleID": self.vehicleID,
            "title": self.title,
            "conformanceStatus": self.conformanceStatus,
            "signs": self.signs,
            "checkpointID": self.checkpointID
        }

    def update(self, db):
        db.collection(u'checkpointInspections').document(self.id).set(self.to_dict())
