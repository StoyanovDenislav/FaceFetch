-- Database schema for a facial recognition system

CREATE TABLE persons (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);

CREATE INDEX ix_persons_name ON persons(name);

CREATE TABLE face_encodings (
    id SERIAL PRIMARY KEY,
    person_id INT NOT NULL,
    encoding BYTEA NOT NULL,
    CONSTRAINT fk_face_encodings_persons FOREIGN KEY (person_id)
        REFERENCES persons(id) ON DELETE CASCADE
);

CREATE INDEX ix_face_encodings_person_id ON face_encodings(person_id);

CREATE TABLE recognition_logs (
    id BIGSERIAL PRIMARY KEY,
    person_id INT,
    confidence NUMERIC(5,4) CHECK (confidence >= 0 AND confidence <= 1),
    image_path VARCHAR(500),
    recognized_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_recognition_logs_persons FOREIGN KEY (person_id)
        REFERENCES persons(id) ON DELETE SET NULL
);

CREATE INDEX ix_recognition_logs_person_id ON recognition_logs(person_id);
CREATE INDEX ix_recognition_logs_recognized_at ON recognition_logs(recognized_at);