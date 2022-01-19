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

const SectionTitle = styled.h3`
  font-family: "Roboto", sans-serif;
  font-size: 1rem;
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
      <Box sx={{ width: "100%" }}>
        <Typography>
          Named Entity Recognition (NER), a sub-task of natural language
          processing (NLP), seeks to identify entities within free text and
          annotate them with categories (e.g. peoples, places, or
          classifications). This demo uses an algorithmic approach to complete
          the NER task. Specifically, a method that leverages{" "}
          <a href="https://en.wikipedia.org/wiki/Trie">tries</a> filled with
          dictionary entries to conduct a fuzzy search on input text.
        </Typography>
        <SectionTitle>How-to-Use</SectionTitle>
        <Box
          sx={{
            fontFamily: "Roboto, sans-serif",
            lineHeight: 1.5,
            fontSize: ".9rem",
            mb: 4,
          }}
        >
          <ol>
            <li>Type input text into text area</li>
            <li>Click "Annotate"</li>
            <li>Annotations will appear below the "Annotations" header</li>
          </ol>
        </Box>
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
            mb: 4,
          }}
          variant="contained"
          startIcon={<PlayArrowIcon />}
          onClick={() => annotateText(inputText)}
        >
          Annotate
        </Button>
      </Box>
      <Divider sx={{ mb: 4 }} />
      <SectionTitle>Annotated Text</SectionTitle>
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
