class Vehicle:
    """
    @authors: Charlie Powell, Benjamin Sanati
    """
    def __init__(self, id, timestamp, title, conformanceStatus, location, lastVehicleInspectionID):
        self.id = id
        self.timestamp = timestamp
        self.title = title
        self.conformanceStatus = conformanceStatus
        self.location = location
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
            data["lastVehicleInspectionID"]
        )

    def to_dict(self):
        return {
            "location": self.location,
            "timestamp": self.timestamp,
            "title": self.title,
            "conformanceStatus": self.conformanceStatus,
            "lastVehicleInspectionID": self.lastVehicleInspectionID
        }

    def update(self, db):
        db.collection(u'vehicles').document(self.id).set(self.to_dict())


class Checkpoint:
    """
    @authors: Charlie Powell, Benjamin Sanati
    """
    def __init__(self, id, timestamp, index, vehicleID, title, prompt, signs, conformanceStatus, lastVehicleInspectionID, lastVehicleInspectionResult, lastVehicleRemediationID, remediationStatus, captureType):
        self.id = id
        self.timestamp = timestamp
        self.index = index
        self.vehicleID = vehicleID
        self.title = title
        self.prompt = prompt
        self.signs = signs
        self.conformanceStatus = conformanceStatus
        self.lastVehicleInspectionID = lastVehicleInspectionID
        self.lastVehicleInspectionResult = lastVehicleInspectionResult
        self.lastVehicleRemediationID = lastVehicleRemediationID
        self.remediationStatus = remediationStatus
        self.captureType = captureType

    @staticmethod
    def from_doc(doc):
        data = doc.to_dict()

        return Checkpoint(
            doc.id,
            data["timestamp"],
            data["index"],
            data["vehicleID"],
            data["title"],
            data["prompt"],
            data["signs"],
            data["conformanceStatus"],
            data["lastVehicleInspectionID"],
            data["lastVehicleInspectionResult"],
            data["lastVehicleRemediationID"],
            data["remediationStatus"],
            data["captureType"]
        )

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "vehicleID": self.vehicleID,
            "index": self.index,
            "title": self.title,
            "prompt": self.prompt,
            "conformanceStatus": self.conformanceStatus,
            "lastVehicleInspectionID": self.lastVehicleInspectionID,
            "lastVehicleInspectionResult": self.lastVehicleInspectionResult,
            "lastVehicleRemediationID": self.lastVehicleRemediationID,
            "signs": self.signs,
            "remediationStatus": self.remediationStatus,
            "captureType": self.captureType
        }

    def update(self, db):
        db.collection(u'checkpoints').document(self.id).set(self.to_dict())


class vehicleInspection:
    """
    @authors: Charlie Powell, Benjamin Sanati
    """
    def __init__(self, id, timestamp, vehicleID, processingStatus, conformanceStatus, location):
        self.id = id
        self.timestamp = timestamp
        self.vehicleID = vehicleID
        self.processingStatus = processingStatus
        self.conformanceStatus = conformanceStatus
        self.location = location

    @staticmethod
    def from_doc(doc):
        data = doc.to_dict()

        return vehicleInspection(
            doc.id,
            data["timestamp"],
            data["vehicleID"],
            data["processingStatus"],
            data["conformanceStatus"],
            data["location"]
        )

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "vehicleID": self.vehicleID,
            "processingStatus": self.processingStatus,
            "conformanceStatus": self.conformanceStatus,
            "location": self.location
        }

    def update(self, db):
        db.collection(u'vehicleInspections').document(self.id).set(self.to_dict())


class CheckpointInspection:
    """
    @authors: Charlie Powell, Benjamin Sanati
    """
    def __init__(self, id, index, timestamp, vehicleID, title, conformanceStatus, signs, checkpointID, vehicleInspectionID, captureType):
        self.id = id
        self.index = index
        self.timestamp = timestamp
        self.vehicleID = vehicleID
        self.checkpointID = checkpointID
        self.title = title
        self.conformanceStatus = conformanceStatus
        self.signs = signs
        self.vehicleInspectionID = vehicleInspectionID
        self.captureType = captureType

    @staticmethod
    def from_doc(doc):
        data = doc.to_dict()

        return CheckpointInspection(
            doc.id,
            data["index"],
            data["timestamp"],
            data["vehicleID"],
            data["title"],
            data["conformanceStatus"],
            data["signs"],
            data["checkpointID"],
            data["vehicleInspectionID"],
            data["captureType"]
        )

    def to_dict(self):
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "vehicleID": self.vehicleID,
            "title": self.title,
            "conformanceStatus": self.conformanceStatus,
            "signs": self.signs,
            "checkpointID": self.checkpointID,
            "vehicleInspectionID": self.vehicleInspectionID,
            "captureType": self.captureType
        }

    def update(self, db):
        db.collection(u'checkpointInspections').document(self.id).set(self.to_dict())
