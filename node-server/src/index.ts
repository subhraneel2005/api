import express from "express";
import axios from "axios";
import cors from "cors";
import "dotenv/config";

const app = express();
app.use(express.json());
app.use(cors());

const PORT = process.env.PORT;
const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL;

app.post("/api/predict", async (req, res) => {
  try {
    const response = await axios.post(
      `${PYTHON_SERVICE_URL}/predict`,
      req.body
    );
    return res.json(response.data);
  } catch (err: any) {
    console.error(
      "Error calling Python service:",
      err.response?.data || err.message
    );

    // Forward the actual error from Python service
    if (err.response) {
      return res.status(err.response.status).json({
        error:
          err.response.data?.detail ||
          err.response.data ||
          "Python service error",
        status: err.response.status,
      });
    }

    // Network or connection error
    return res.status(500).json({
      error: "Failed to connect to prediction service",
      details: err.message,
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port: ${PORT}`);
});
