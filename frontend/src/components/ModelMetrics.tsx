import React, { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  Typography,
  Box,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  SelectChangeEvent,
} from "@mui/material";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
} from "recharts";
import axios from "axios";

interface ParameterMetrics {
  mse: number;
  mae: number;
}

interface ModelMetrics {
  [key: string]: ParameterMetrics;
}

interface ActualVsPredicted {
  actual: number[];
  predicted: number[];
  timestamps: string[];
}

const ModelMetrics: React.FC = () => {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedParameter, setSelectedParameter] = useState<string>("pm25");
  const [actualVsPredicted, setActualVsPredicted] =
    useState<ActualVsPredicted | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await axios.get(
          "http://localhost:5000/api/model/metrics"
        );
        setMetrics(response.data);
        if (Object.keys(response.data).length > 0) {
          setSelectedParameter(Object.keys(response.data)[0]);
        }
      } catch (err) {
        setError("Failed to fetch model metrics");
        console.error("Error fetching metrics:", err);
      }
    };

    fetchMetrics();
  }, []);

  useEffect(() => {
    const fetchActualVsPredicted = async () => {
      if (!selectedParameter) return;

      try {
        const response = await axios.get(
          `http://localhost:5000/api/model/actual-vs-predicted/Kyiv/${selectedParameter}`
        );
        setActualVsPredicted(response.data);
      } catch (err) {
        console.error("Error fetching actual vs predicted data:", err);
      }
    };

    fetchActualVsPredicted();
  }, [selectedParameter]);

  const handleParameterChange = (event: SelectChangeEvent) => {
    setSelectedParameter(event.target.value);
  };

  if (error) {
    return (
      <Card>
        <CardContent>
          <Typography color="error">{error}</Typography>
        </CardContent>
      </Card>
    );
  }

  if (!metrics) {
    return (
      <Card>
        <CardContent>
          <Typography>Loading metrics...</Typography>
        </CardContent>
      </Card>
    );
  }

  // Prepare data for charts
  const mseData = Object.entries(metrics).map(([parameter, values]) => ({
    parameter: parameter.toUpperCase(),
    value: values.mse,
  }));

  const maeData = Object.entries(metrics).map(([parameter, values]) => ({
    parameter: parameter.toUpperCase(),
    value: values.mae,
  }));

  // Prepare data for actual vs predicted chart
  const actualVsPredictedData = actualVsPredicted
    ? actualVsPredicted.timestamps.map((timestamp, index) => ({
        timestamp,
        actual: actualVsPredicted.actual[index],
        predicted: actualVsPredicted.predicted[index],
      }))
    : [];

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          Model Performance Metrics
        </Typography>
        <Box
          sx={{
            display: "flex",
            flexDirection: { xs: "column", md: "row" },
            gap: 3,
          }}
        >
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" gutterBottom>
              Mean Squared Error (MSE)
            </Typography>
            <div style={{ width: "100%", height: 300 }}>
              <ResponsiveContainer>
                <BarChart data={mseData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="parameter" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Box>
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" gutterBottom>
              Mean Absolute Error (MAE)
            </Typography>
            <div style={{ width: "100%", height: 300 }}>
              <ResponsiveContainer>
                <BarChart data={maeData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="parameter" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Box>
        </Box>

        {/* <Box sx={{ mt: 4 }}>
          <Typography variant="h6" gutterBottom>
            Actual vs Predicted Values
          </Typography>
          <FormControl sx={{ mb: 2, minWidth: 200 }}>
            <InputLabel>Parameter</InputLabel>
            <Select
              value={selectedParameter}
              label="Parameter"
              onChange={handleParameterChange}
            >
              {Object.keys(metrics).map((param) => (
                <MenuItem key={param} value={param}>
                  {param.toUpperCase()}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <div style={{ width: "100%", height: 400 }}>
            <ResponsiveContainer>
              <LineChart data={actualVsPredictedData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="timestamp"
                  tick={{ fontSize: 12 }}
                  angle={-45}
                  textAnchor="end"
                />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#8884d8"
                  name="Actual"
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#82ca9d"
                  name="Predicted"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Box> */}

        <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>
          Detailed Metrics
        </Typography>
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: {
              xs: "1fr",
              sm: "repeat(2, 1fr)",
              md: "repeat(3, 1fr)",
            },
            gap: 2,
          }}
        >
          {Object.entries(metrics).map(([parameter, values]) => (
            <Card variant="outlined" key={parameter}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {parameter.toUpperCase()}
                </Typography>
                <Typography variant="body2">
                  MSE: {values.mse.toFixed(4)}
                </Typography>
                <Typography variant="body2">
                  MAE: {values.mae.toFixed(4)}
                </Typography>
              </CardContent>
            </Card>
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ModelMetrics;
