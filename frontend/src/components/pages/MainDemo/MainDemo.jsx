import { React, useState } from "react";
import styled from "styled-components";

import { exampleText, mockAnnotations } from "./constants";

import Container from "@mui/material/Container";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import AnnotatedText from "./AnnotatedText";
import { Button, Box, TextField, Typography, Divider } from "@mui/material";

const PageTitle = styled.h2`
  font-family: "Roboto", sans-serif;
  font-size: 1.5rem;
`;

const inputStyles = {
  style: {
    fontSize: ".9rem",
    fontFamily: "Roboto mono",
    padding: ".5rem",
  },
};

function MainDemo() {
  const [inputText, setInputText] = useState(exampleText);
  const [annotatedText, setAnnotatedText] = useState(null);

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  const annotateText = (text) => {
    // TODO eventually do some API call or something
    setAnnotatedText(text);
  };

  return (
    <Container>
      <PageTitle>Main Demo</PageTitle>
      <Box sx={{ width: "100%", height: 150 }}>
        {" "}
        <Typography>Explanation of demo goes here.</Typography>
      </Box>
      <Box
        sx={{
          "& .MuiTextField-root": { maxWidth: "100%", mb: 2 },
        }}
      >
        <TextField
          fullWidth
          label="Text Input"
          multiline
          rows={6}
          onChange={handleInputChange}
          inputProps={inputStyles}
          defaultValue={inputText}
          variant="standard"
        />
        <Button
          sx={{
            mb: 2,
          }}
          variant="contained"
          startIcon={<PlayArrowIcon />}
          onClick={() => annotateText(inputText)}
        >
          Annotate
        </Button>
      </Box>
      <Divider sx={{ mb: 4 }} />
      {annotatedText && (
        <AnnotatedText
          outputText={annotatedText}
          annotations={mockAnnotations}
        />
      )}
    </Container>
  );
}

export default MainDemo;
